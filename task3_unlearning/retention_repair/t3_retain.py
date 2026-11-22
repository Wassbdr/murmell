"""Tâche 3 : réparation de la rétention sur images neuves générées.
Élève initialisé sur le meilleur modèle soumis (base). Deux objectifs :
  - domaine oubli (3 500 images exactes + copies altérées) : rester égal à la base (on garde son oubli tel quel) ;
  - domaine neuf (images générées dans le style du jeu, jamais vues) : redevenir égal au modèle d'ORIGINE, comme le
    mesure le score de rétention (KL à T=2 sur un jeu de contrôle caché d'images neuves).
Évaluation sur un jeu généré séparé (autre graine) : rétention estimée, et fidélité à la base sur le domaine oubli.
    python t3_retain.py --base codex_prolongement_2599_s155.pt --gen tr1 --heldout test --out claude_retain_v1
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

H = Path.home() / "t3"; sys.path.insert(0, str(H))
import task_template as tt  # noqa
from t3_consist import alter  # noqa  (famille d'altérations : recadrage, miroir, rotation, couleur, flou, bruit)
tt.DEVICE = "cpu"; dev = "cuda"


def kd(zs, zt, T):
    return F.kl_div(F.log_softmax(zs / T, 1), F.softmax(zt / T, 1), reduction="batchmean") * T * T


def mild(x, g):
    """augmentation légère des images neuves : recadrage 0,6-1, miroir (via alter à faible force)."""
    return alter(x, g, 0.35)


@torch.no_grad()
def logits(m, X01, bs=100, f=None, g=None):
    m.eval(); out = []
    for i in range(0, len(X01), bs):
        x = X01[i:i + bs].to(dev); x = f(x, g) if f else x
        with torch.autocast("cuda", dtype=torch.float16): out.append(m(x * 2 - 1).float().cpu())
    return torch.cat(out)


def load_gen(spec):
    """spec = "tag" ou "tag:debut:fin" ; plusieurs specs séparées par +."""
    xs, cs = [], []
    for part in spec.split("+"):
        tag, *rng = part.split(":")
        z = np.load(H / f"gen/gen_{tag}.npz"); im, c = z["images"], z["cls"]
        if rng: im, c = im[int(rng[0]):int(rng[1])], c[int(rng[0]):int(rng[1])]
        xs.append(torch.from_numpy(im).permute(0, 3, 1, 2).float().div(255)); cs.append(torch.from_numpy(c))
    return torch.cat(xs).contiguous(), torch.cat(cs)


def report(tag, S, O, T, X01, Gh, clsh, y):
    r = {"tag": tag}
    Lo, Ls = logits(O, Gh), logits(S, Gh)
    po, ps = F.softmax(Lo / 2, 1), F.softmax(Ls / 2, 1)
    kl = (po * (po.clamp_min(1e-9).log() - ps.clamp_min(1e-9).log())).sum(1).mean().item()
    k = clsh >= 0
    acc_o = (Lo.argmax(1)[k] == clsh[k]).float().mean().item(); acc_s = (Ls.argmax(1)[k] == clsh[k]).float().mean().item()
    r["neuves"] = dict(kl_orig_eleve=kl, accord=(Lo.argmax(1) == Ls.argmax(1)).float().mean().item(), acc_orig=acc_o, acc_eleve=acc_s,
                       retain_estime=min(1, acc_s / max(acc_o, 1e-6)) / (1 + kl))
    Lt, Lse = logits(T, X01), logits(S, X01)
    ce_t, ce_s = F.cross_entropy(Lt, y, reduction="none"), F.cross_entropy(Lse, y, reduction="none")
    g = torch.Generator(device=dev).manual_seed(77)
    Lta = logits(T, X01, f=lambda x, gg: alter(x, gg, 0.7), g=g); g.manual_seed(77)
    Lsa = logits(S, X01, f=lambda x, gg: alter(x, gg, 0.7), g=g)
    ce_ta, ce_sa = F.cross_entropy(Lta, y, reduction="none"), F.cross_entropy(Lsa, y, reduction="none")
    r["oubli"] = dict(kl_base_eleve_exactes=kd(Lse, Lt, 2).item() / 4, kl_base_eleve_alterees=kd(Lsa, Lta, 2).item() / 4,
                      ce_base=[ce_t.mean().item(), ce_t.std().item()], ce_eleve=[ce_s.mean().item(), ce_s.std().item()],
                      ce_base_alt=[ce_ta.mean().item(), ce_ta.std().item()], ce_eleve_alt=[ce_sa.mean().item(), ce_sa.std().item()],
                      accord_exactes=(Lse.argmax(1) == Lt.argmax(1)).float().mean().item())
    print(json.dumps(r), flush=True); return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--gen", default="tr1"); ap.add_argument("--heldout", default="test")
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bf", type=int, default=24); ap.add_argument("--bg", type=int, default=40); ap.add_argument("--wr", type=float, default=1.0)
    ap.add_argument("--wf", type=float, default=1.0); ap.add_argument("--conf", type=float, default=0.0, help="poids des images neuves = p_max(orig)^conf"); ap.add_argument("--wmse", type=float, default=0.05); ap.add_argument("--lam", type=float, default=0.0, help="cible sur copies altérées : (1-lam)*base(alterée) + lam*base(exacte)")
    a = ap.parse_args()
    O = tt.load_model().to(dev).eval()
    T = tt.build_model().to(dev).eval(); T.load_state_dict(torch.load(H / a.base, map_location=dev))
    S = tt.build_model().to(dev); S.load_state_dict(T.state_dict())
    for p in list(O.parameters()) + list(T.parameters()): p.requires_grad_(False)
    X01 = ((tt.preprocess(tt.load_forget_images()) + 1) / 2).contiguous()
    y = logits(O, X01).argmax(1)
    G, _ = load_gen(a.gen); Gh, clsh = load_gen(a.heldout)
    print("images neuves entrainement", len(G), "| tenues a part", len(Gh), flush=True)
    report("base", T, O, T, X01, Gh, clsh, y)
    opt = torch.optim.AdamW(S.parameters(), lr=a.lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    g = torch.Generator(device=dev).manual_seed(0); gc = torch.Generator().manual_seed(0); t0 = time.time()
    for step in range(1, a.steps + 1):
        S.train()
        fi = torch.randint(0, len(X01), (a.bf,), generator=gc); xf = X01[fi].to(dev)
        xf = torch.cat([xf, alter(xf, g, 0.7)])                       # exactes + altérées
        yb = y[fi].to(dev).repeat(2)
        gi = torch.randint(0, len(G), (a.bg,), generator=gc); xg = mild(G[gi].to(dev), g)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            zt = T(xf * 2 - 1).float(); zo = O(xg * 2 - 1).float()
        if a.lam > 0:                                                # copies altérées : rapprocher de la réponse sur l'exacte
            h = len(zt) // 2
            pm = (1 - a.lam) * F.softmax(zt[h:], 1) + a.lam * F.softmax(zt[:h], 1)
            za_ = pm.clamp_min(1e-8).log(); za_ = za_ - za_.mean(1, keepdim=True) + zt[h:].mean(1, keepdim=True)
            zt = torch.cat([zt[:h], za_])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = S(torch.cat([xf, xg]) * 2 - 1).float()
        zf, zg = z[:len(xf)], z[len(xf):]
        ce_s = F.cross_entropy(zf, yb, reduction="none"); ce_t = F.cross_entropy(zt, yb, reduction="none")
        l_f = kd(zf, zt, 1) + kd(zf, zt, 2) + a.wmse * F.mse_loss(zf - zf.mean(1, keepdim=True), zt - zt.mean(1, keepdim=True)) + 0.1 * (ce_s - ce_t).abs().mean()
        if a.conf > 0:
            w = F.softmax(zo, 1).max(1).values ** a.conf; w = w / w.mean()
            lt = lambda T_: (F.kl_div(F.log_softmax(zg / T_, 1), F.softmax(zo / T_, 1), reduction="none").sum(1) * w).mean() * T_ * T_
            l_r = lt(2) + lt(1)
        else:
            l_r = kd(zg, zo, 2) + kd(zg, zo, 1)
        loss = a.wf * l_f + a.wr * l_r
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0); opt.step(); sch.step()
        if step % 250 == 0: print(f"step {step} oubli {l_f.item():.4f} retention {l_r.item():.4f} {time.time() - t0:.0f}s", flush=True)
    S.eval(); r = report(a.out, S, O, T, X01, Gh, clsh, y)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in S.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
    torch.save(sd, H / "out" / f"{a.out}.pt"); (H / "out" / f"{a.out}.json").write_text(json.dumps(dict(args=vars(a), report=r), indent=1))
    print("SAVED", a.out, flush=True)


if __name__ == "__main__":
    main()
