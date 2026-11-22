"""Distille deux modeles desappris et leur soupe dans un seul modele.

La cible sur le domaine d'oubli est une moyenne arithmetique des distributions,
comme dans la metrique de prediction officielle. La retention utilise le modele
d'origine sur des images generees neuves. Aucune contrainte de coherence entre
images exactes et alterees n'est ajoutee.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

H = Path.home() / "t3"
sys.path.insert(0, str(H))
import task_template as tt
from t3_consist import alter
from t3_retain import load_gen, logits, mild, report

tt.DEVICE = "cpu"
DEV = "cuda"


def kl_prob(student, target, temperature):
    return F.kl_div(
        F.log_softmax(student / temperature, 1), target,
        reduction="batchmean",
    ) * temperature * temperature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/claude_soupe_rep.pt")
    ap.add_argument("--teachers", default="out/claude_rep_lina_I_E4d05_tr03.pt,out/claude_rep_lina_I_E4d08.pt")
    ap.add_argument("--ensemble", type=float, default=0.67, help="poids total des deux enseignants independants")
    ap.add_argument("--gen", default="tr1+test:0:650")
    ap.add_argument("--heldout", default="test:650:852")
    ap.add_argument("--out", default="kimi_ensemble_soupe")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bf", type=int, default=16)
    ap.add_argument("--bg", type=int, default=32)
    ap.add_argument("--wr", type=float, default=1.5)
    ap.add_argument("--wf", type=float, default=4.0)
    ap.add_argument("--conf", type=float, default=2.0)
    ap.add_argument("--wmse", type=float, default=0.2)
    args = ap.parse_args()

    original = tt.load_model().to(DEV).eval()
    base = tt.build_model().to(DEV).eval()
    base.load_state_dict(torch.load(H / args.base, map_location=DEV, weights_only=True))
    student = tt.build_model().to(DEV)
    student.load_state_dict(base.state_dict())
    teachers = []
    for path in args.teachers.split(","):
        model = tt.build_model().to(DEV).eval()
        model.load_state_dict(torch.load(H / path, map_location=DEV, weights_only=True))
        teachers.append(model)
    for model in [original, base] + teachers:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    x_forget = ((tt.preprocess(tt.load_forget_images()) + 1) / 2).contiguous()
    y = logits(original, x_forget).argmax(1)
    generated, _ = load_gen(args.gen)
    heldout, heldout_classes = load_gen(args.heldout)
    print("images neuves", len(generated), "tenues a part", len(heldout), flush=True)
    report("base", base, original, base, x_forget, heldout, heldout_classes, y)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps, pct_start=0.1)
    generator = torch.Generator(device=DEV).manual_seed(0)
    cpu_generator = torch.Generator().manual_seed(0)
    start = time.time()

    for step in range(1, args.steps + 1):
        student.train()
        ids = torch.randint(0, len(x_forget), (args.bf,), generator=cpu_generator)
        xf = x_forget[ids].to(DEV)
        xf = torch.cat([xf, alter(xf, generator, 0.7)])
        labels = y[ids].to(DEV).repeat(2)
        gi = torch.randint(0, len(generated), (args.bg,), generator=cpu_generator)
        xg = mild(generated[gi].to(DEV), generator)

        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            zb = base(xf * 2 - 1).float()
            teacher_logits = [model(xf * 2 - 1).float() for model in teachers]
            zo = original(xg * 2 - 1).float()
        independent = torch.stack(teacher_logits).mean(0)
        target_logits = (1 - args.ensemble) * zb + args.ensemble * independent
        ce_target = (1 - args.ensemble) * F.cross_entropy(zb, labels, reduction="none")
        ce_target = ce_target + args.ensemble * torch.stack([
            F.cross_entropy(z, labels, reduction="none") for z in teacher_logits
        ]).mean(0)

        with torch.autocast(DEV, dtype=torch.bfloat16):
            z = student(torch.cat([xf, xg]) * 2 - 1).float()
        zf, zg = z[:len(xf)], z[len(xf):]
        losses = []
        for temperature in (1, 2):
            qb = F.softmax(zb / temperature, 1)
            qi = torch.stack([F.softmax(z / temperature, 1) for z in teacher_logits]).mean(0)
            q = (1 - args.ensemble) * qb + args.ensemble * qi
            losses.append(kl_prob(zf, q, temperature))
        ce_student = F.cross_entropy(zf, labels, reduction="none")
        l_forget = sum(losses) + args.wmse * F.mse_loss(zf, target_logits)
        l_forget = l_forget + 0.1 * (ce_student - ce_target).abs().mean()

        if args.conf > 0:
            weights = F.softmax(zo, 1).max(1).values ** args.conf
            weights = weights / weights.mean()
            def retention_at(temperature):
                per_sample = F.kl_div(
                    F.log_softmax(zg / temperature, 1), F.softmax(zo / temperature, 1),
                    reduction="none",
                ).sum(1)
                return (per_sample * weights).mean() * temperature * temperature
            l_retain = retention_at(2) + retention_at(1)
        else:
            from t3_retain import kd
            l_retain = kd(zg, zo, 2) + kd(zg, zo, 1)

        loss = args.wf * l_forget + args.wr * l_retain
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 250 == 0:
            print(f"step {step} oubli {l_forget.item():.4f} retention {l_retain.item():.4f} {time.time() - start:.0f}s", flush=True)

    student.eval()
    result = report(args.out, student, original, base, x_forget, heldout, heldout_classes, y)
    state = {k: v.detach().float().cpu().contiguous() for k, v in student.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    torch.save(state, H / "out" / f"{args.out}.pt")
    import json
    (H / "out" / f"{args.out}.json").write_text(json.dumps({"args": vars(args), "report": result}, indent=1))
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
