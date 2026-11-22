"""Distille une correction des mauvaises classes a perte quasi constante."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

H = Path.home() / "t3"
sys.path.insert(0, str(H))
import task_template as tt
from t3_consist import alter
from t3_retain import kd, load_gen, logits, mild, report

tt.DEVICE = "cpu"
DEV = "cuda"


def conditional_target(z, labels, pg_alt, alpha, temperature):
    p = F.softmax(z / temperature, 1)
    mask = F.one_hot(labels, num_classes=100).bool()
    py = p.gather(1, labels[:, None])
    alt = p.masked_fill(mask, 0)
    alt = alt / alt.sum(1, keepdim=True).clamp_min(1e-12)
    alt = (1 - alpha) * alt + alpha * pg_alt
    q = alt * (1 - py)
    q.scatter_add_(1, labels[:, None], py)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/kimi_proxy_soup50.pt")
    ap.add_argument("--pg", default="kimi_pg.pt")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="kimi_conditional_005")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bf", type=int, default=16)
    ap.add_argument("--bg", type=int, default=32)
    ap.add_argument("--wf", type=float, default=4.0)
    ap.add_argument("--wr", type=float, default=1.5)
    ap.add_argument("--wmse", type=float, default=0.05)
    ap.add_argument("--wce", type=float, default=0.1)
    ap.add_argument("--nblocks", type=int, default=0)
    ap.add_argument("--shift", type=float, default=0.0, help="suppression du logit vrai sur les vues alterees")
    ap.add_argument("--clean_weight", type=float, default=1.0)
    ap.add_argument("--altered_weight", type=float, default=1.0)
    ap.add_argument("--recipe_augment", action="store_true")
    args = ap.parse_args()

    original = tt.load_model().to(DEV).eval()
    base = tt.build_model().to(DEV).eval()
    base.load_state_dict(torch.load(H / args.base, map_location=DEV, weights_only=True))
    student = tt.build_model().to(DEV)
    student.load_state_dict(base.state_dict())
    if args.nblocks > 0:
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        train_params = [p for block in student.blocks[-args.nblocks:] for p in block.parameters()]
        train_params += list(student.norm.parameters()) + list(student.head.parameters())
        for parameter in train_params:
            parameter.requires_grad_(True)
    else:
        train_params = list(student.parameters())
    for model in (original, base):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    target = torch.load(H / args.pg, map_location="cpu", weights_only=True)
    pg = target["pg"].to(DEV)
    y_cpu = target["y"].long()
    y = y_cpu.to(DEV)
    mask = F.one_hot(y, num_classes=100).bool()
    pg_alt = pg.masked_fill(mask, 0)
    pg_alt = pg_alt / pg_alt.sum(1, keepdim=True).clamp_min(1e-12)

    x_forget = ((tt.preprocess(tt.load_forget_images()) + 1) / 2).contiguous()
    assert torch.equal(y_cpu, logits(original, x_forget).argmax(1).cpu())
    generated, _ = load_gen("tr1+test:0:650")
    heldout, classes = load_gen("test:650:852")
    report("base", base, original, base, x_forget, heldout, classes, y_cpu)

    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps, pct_start=0.1)
    if args.recipe_augment:
        sys.path.insert(0, "/workspace/unlearning")
        import unlearn as recipe
    generator = torch.Generator(device=DEV).manual_seed(0)
    cpu_generator = torch.Generator().manual_seed(0)
    start = time.time()
    for step in range(1, args.steps + 1):
        student.train()
        ids = torch.randint(0, len(x_forget), (args.bf,), generator=cpu_generator)
        xf = x_forget[ids].to(DEV)
        if args.recipe_augment:
            strength = torch.full((args.bf,), 0.6, device=DEV)
            xf = torch.cat([xf, recipe.augment(xf, strength, generator)])
        else:
            xf = torch.cat([xf, alter(xf, generator, 0.7)])
        labels = y[ids].to(DEV).repeat(2)
        targets = pg_alt[ids].to(DEV).repeat(2, 1)
        gi = torch.randint(0, len(generated), (args.bg,), generator=cpu_generator)
        xg = mild(generated[gi].to(DEV), generator)
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            zt = base(xf * 2 - 1).float()
            zo = original(xg * 2 - 1).float()
        with torch.autocast(DEV, dtype=torch.bfloat16):
            z = student(torch.cat([xf, xg]) * 2 - 1).float()
        zf, zg = z[:len(xf)], z[len(xf):]
        losses = []
        if args.shift > 0:
            target_logits = zt.clone()
            altered_rows = torch.arange(args.bf, 2 * args.bf, device=DEV)
            target_logits[altered_rows, labels[args.bf:]] -= args.shift
            for temperature in (1, 2):
                if args.clean_weight != 1.0:
                    clean_loss = kd(zf[:args.bf], zt[:args.bf], temperature)
                    altered_loss = kd(zf[args.bf:], target_logits[args.bf:], temperature)
                    losses.append(args.clean_weight * clean_loss + args.altered_weight * altered_loss)
                else:
                    losses.append(kd(zf, target_logits, temperature))
            ce_teacher = F.cross_entropy(target_logits, labels, reduction="none")
        else:
            for temperature in (1, 2):
                q = conditional_target(zt, labels, targets, args.alpha, temperature)
                losses.append(F.kl_div(F.log_softmax(zf / temperature, 1), q, reduction="batchmean") * temperature * temperature)
            ce_teacher = F.cross_entropy(zt, labels, reduction="none")
        ce_student = F.cross_entropy(zf, labels, reduction="none")
        l_forget = sum(losses) + args.wmse * F.mse_loss(zf, zt) + args.wce * (ce_student - ce_teacher).abs().mean()
        weights = F.softmax(zo, 1).max(1).values ** 2
        weights = weights / weights.mean()
        def retention_at(temperature):
            per_sample = F.kl_div(F.log_softmax(zg / temperature, 1), F.softmax(zo / temperature, 1), reduction="none").sum(1)
            return (per_sample * weights).mean() * temperature * temperature
        l_retain = retention_at(2) + retention_at(1)
        loss = args.wf * l_forget + args.wr * l_retain
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        scheduler.step()
        if step % 250 == 0:
            print(f"step {step} oubli {l_forget.item():.4f} retention {l_retain.item():.4f} {time.time() - start:.0f}s", flush=True)

    student.eval()
    result = report(args.out, student, original, base, x_forget, heldout, classes, y_cpu)
    state = {k: v.detach().float().cpu().contiguous() for k, v in student.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    torch.save(state, H / "out" / f"{args.out}.pt")
    (H / "out" / f"{args.out}.json").write_text(json.dumps({"args": vars(args), "report": result}, indent=1))
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
