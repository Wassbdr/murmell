"""Tâche 3, candidat "cohérence sous altérations".
Base = meilleur modèle soumis (professeur figé). L'élève part de la base et apprend à donner, sur toute copie altérée
d'une image à oublier (recadrage, miroir, petite rotation, couleur, flou, bruit : famille décrite dans l'historique du
dépôt du sujet), la réponse que la base donne sur l'image exacte. Ancrages : image exacte (réponse inchangée) et entrées
hors famille (recadrages extrêmes, mélanges, retournements) où l'élève doit rester égal à la base.
    python t3_consist.py --base codex_moments_globaux_2458.pt --out claude_consist_v1 --steps 800
"""
import argparse, json, math, sys, time, torch, torch.nn.functional as F
from pathlib import Path
H = Path.home() / "t3"; sys.path.insert(0, str(H))
import task_template as tt
tt.DEVICE = "cpu"; dev = "cuda"


def U(a, b, n, g): return torch.empty(n, device=dev).uniform_(a, b, generator=g)


def blur1(x, sig):
    k = int(2 * round(3 * sig) + 1)
    if k < 3: return x
    a = torch.arange(k, device=x.device) - k // 2; w = torch.exp(-a ** 2 / (2 * sig ** 2)); w = (w / w.sum()).to(x.dtype)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def alter(x, g, strength=1.0):
    """x [B,3,224,224] dans [0,1] sur GPU. Chaque transformation est tirée indépendamment ; force aléatoire par image."""
    B = len(x)
    # recadrage + rotation + miroir en une seule grille affine
    s = U(0.30, 1.0, B, g) ** 0.5 if strength >= 1 else U(0.55, 1.0, B, g) ** 0.5
    s = torch.where(torch.rand(B, device=dev, generator=g) < 0.85, s, torch.ones_like(s))
    r = torch.exp(U(math.log(3 / 4), math.log(4 / 3), B, g)) ** 0.5
    sx, sy = (s * r).clamp(max=1), (s / r).clamp(max=1)
    tx, ty = U(-1, 1, B, g) * (1 - sx), U(-1, 1, B, g) * (1 - sy)
    th = U(-15, 15, B, g) * math.pi / 180 * (torch.rand(B, device=dev, generator=g) < 0.6)
    fl = torch.where(torch.rand(B, device=dev, generator=g) < 0.5, -1.0, 1.0)
    A = torch.stack([torch.stack([sx * th.cos() * fl, -sy * th.sin(), tx], 1), torch.stack([sx * th.sin() * fl, sy * th.cos(), ty], 1)], 1)
    pm = ["zeros", "reflection", "border"][int(torch.randint(0, 3, (1,), device=dev, generator=g))]
    x = F.grid_sample(x, F.affine_grid(A, x.shape, align_corners=False), mode="bilinear", padding_mode=pm, align_corners=False)
    # couleur : luminosité, contraste, saturation, teinte (rotation dans le plan chromatique), parfois gris
    on = (torch.rand(B, 4, device=dev, generator=g) < 0.7).float()
    k = 0.45 * strength
    b = 1 + U(-k, k, B, g) * on[:, 0]; c = 1 + U(-k, k, B, g) * on[:, 1]; sa = 1 + U(-k, k, B, g) * on[:, 2]; hu = U(-0.08, 0.08, B, g) * on[:, 3] * 2 * math.pi * strength
    x = x * b[:, None, None, None]
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * c[:, None, None, None] + m
    yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    t = torch.einsum("ij,bjhw->bihw", yiq, x)
    co, si = hu.cos()[:, None, None], hu.sin()[:, None, None]
    i2 = t[:, 1] * co - t[:, 2] * si; q2 = t[:, 1] * si + t[:, 2] * co
    gray = (torch.rand(B, device=dev, generator=g) < 0.05).float()[:, None, None]
    sa_ = sa[:, None, None] * (1 - gray)
    t = torch.stack([t[:, 0], i2 * sa_, q2 * sa_], 1)
    x = torch.einsum("ij,bjhw->bihw", torch.linalg.inv(yiq), t).clamp(0, 1)
    # flou puis bruit
    out = []
    sig = U(0.3, 2.2 * strength, B, g); bon = torch.rand(B, device=dev, generator=g) < 0.5
    for i in range(B):
        out.append(blur1(x[i:i + 1], float(sig[i])) if bon[i] else x[i:i + 1])
    x = torch.cat(out)
    ns = U(0.0, 0.09 * strength, B, g) * (torch.rand(B, device=dev, generator=g) < 0.5)
    return (x + ns[:, None, None, None] * torch.randn(x.shape, device=dev, generator=g)).clamp(0, 1)


def offfam(x, g):
    """entrées hors famille : là l'élève doit rester égal à la base (protège le comportement général)."""
    B = len(x); k = int(torch.randint(0, 4, (1,), device=dev, generator=g))
    if k == 0:                                    # recadrage extrême
        s = U(0.04, 0.2, B, g) ** 0.5; tx, ty = U(-1, 1, B, g) * (1 - s), U(-1, 1, B, g) * (1 - s)
        A = torch.stack([torch.stack([s, torch.zeros_like(s), tx], 1), torch.stack([torch.zeros_like(s), s, ty], 1)], 1)
        return F.grid_sample(x, F.affine_grid(A, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    if k == 1:                                    # mélange de deux images
        lam = U(0.35, 0.65, B, g)[:, None, None, None]; return lam * x + (1 - lam) * x[torch.randperm(B, device=dev, generator=g)]
    if k == 2:                                    # retournement vertical / quart de tour
        return x.flip(-2) if torch.rand(1, device=dev, generator=g) < 0.5 else torch.rot90(x, 1, (2, 3))
    p = 56; n = 4                                  # puzzle
    t = x.unfold(2, p, p).unfold(3, p, p).reshape(B, 3, n * n, p, p)
    idx = torch.stack([torch.randperm(n * n, device=dev, generator=g) for _ in range(B)])
    t = torch.gather(t, 2, idx[:, None, :, None, None].expand(B, 3, n * n, p, p)).reshape(B, 3, n, n, p, p)
    return t.permute(0, 1, 2, 4, 3, 5).reshape(B, 3, 224, 224)


def kd(zs, zt, T):
    return F.kl_div(F.log_softmax(zs / T, 1), F.softmax(zt / T, 1), reduction="batchmean") * T * T


@torch.no_grad()
def logits(m, X01, f=None, g=None, bs=100):
    m.eval(); out = []
    for i in range(0, len(X01), bs):
        x = X01[i:i + bs].to(dev); x = f(x, g) if f else x
        with torch.autocast("cuda", dtype=torch.float16): out.append(m(x * 2 - 1).float())
    return torch.cat(out)


def report(tag, S, T, O, X01, y, Tc):
    g = torch.Generator(device=dev).manual_seed(999); r = {"tag": tag}
    Lc = logits(S, X01); ce = F.cross_entropy(Lc, y, reduction="none")
    r["clean"] = dict(ce=ce.mean().item(), sd=ce.std().item(), acc=(Lc.argmax(1) == y).float().mean().item(), kd_T2=kd(Lc, Tc, 2).item() / 4)
    for nm, st in (("alt_doux", 0.5), ("alt_fort", 1.0)):
        g.manual_seed(999); La = logits(S, X01, lambda x, gg: alter(x, gg, st), g); ce = F.cross_entropy(La, y, reduction="none")
        r[nm] = dict(ce=ce.mean().item(), sd=ce.std().item(), acc=(La.argmax(1) == y).float().mean().item(), same=(La.argmax(1) == Tc.argmax(1)).float().mean().item(), kd_T2=kd(La, Tc, 2).item() / 4)
    g.manual_seed(5); u = torch.cat([offfam(X01[i:i + 100].to(dev), g).cpu() for i in range(0, 1200, 100)])
    Lu, Tu, Ou = logits(S, u), logits(T, u), logits(O, u)
    r["hors_famille"] = dict(kl_base_eleve=kd(Lu, Tu, 2).item() / 4, kl_orig_eleve=kd(Lu, Ou, 2).item() / 4, kl_orig_base=kd(Tu, Ou, 2).item() / 4,
                             accord_orig_eleve=(Lu.argmax(1) == Ou.argmax(1)).float().mean().item(), accord_orig_base=(Tu.argmax(1) == Ou.argmax(1)).float().mean().item())
    print(json.dumps(r), flush=True); return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--anchor", type=float, default=2.0)
    ap.add_argument("--sp", type=float, default=0.0); ap.add_argument("--wclean", type=float, default=4.0); ap.add_argument("--alt_T", type=float, default=1.0, help="adoucissement de la cible sur copies altérées")
    a = ap.parse_args()
    O = tt.load_model().to(dev).eval()
    T = tt.build_model().to(dev).eval(); T.load_state_dict(torch.load(H / a.base, map_location=dev))
    S = tt.build_model().to(dev); S.load_state_dict(T.state_dict())
    for p in list(O.parameters()) + list(T.parameters()): p.requires_grad_(False)
    X01 = ((tt.preprocess(tt.load_forget_images()) + 1) / 2).contiguous()          # CPU
    y = logits(O, X01).argmax(1); Tc = logits(T, X01)
    report("base", T, T, O, X01, y, Tc)
    opt = torch.optim.AdamW(S.parameters(), lr=a.lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    g = torch.Generator(device=dev).manual_seed(0); gc = torch.Generator().manual_seed(0); t0 = time.time()
    theta0 = [p.detach().clone() for p in S.parameters()] if a.sp > 0 else None
    for step in range(1, a.steps + 1):
        S.train()
        ids = torch.randint(0, len(X01), (a.bs,), generator=gc); x = X01[ids].to(dev); zt = Tc[ids]
        st2 = (0.35, 0.5, 0.7, 1.0)[int(torch.randint(0, 4, (1,), generator=gc))]
        xa = torch.cat([alter(x, g, 0.5), alter(x, g, st2)]); xu = offfam(x, g)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            zu = T(xu * 2 - 1).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = S(torch.cat([x, xa, xu]) * 2 - 1).float()
        zc, za, zs = z[:a.bs], z[a.bs:3 * a.bs], z[3 * a.bs:]
        zta = zt.repeat(2, 1) / a.alt_T
        l_alt = kd(za, zta, 1) + kd(za, zta, 2); l_clean = kd(zc, zt, 1) + kd(zc, zt, 2); l_anc = kd(zs, zu, 1) + kd(zs, zu, 2)
        loss = l_alt + a.wclean * (l_clean + F.mse_loss(zc, zt)) + a.anchor * l_anc
        if a.sp > 0: loss = loss + a.sp * sum(((p - q) ** 2).sum() for p, q in zip(S.parameters(), theta0))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0); opt.step(); sch.step()
        if step % 250 == 0: print(f"step {step} alt {l_alt.item():.4f} clean {l_clean.item():.4f} ancre {l_anc.item():.4f} {time.time() - t0:.0f}s", flush=True)
    S.eval(); r = report(a.out, S, T, O, X01, y, Tc)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in S.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
    (H / "out").mkdir(exist_ok=True); torch.save(sd, H / "out" / f"{a.out}.pt"); (H / "out" / f"{a.out}.json").write_text(json.dumps(dict(args=vars(a), report=r), indent=1))
    print("SAVED", a.out, flush=True)


if __name__ == "__main__":
    main()
