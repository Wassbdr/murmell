"""Copies altérées "à la serveur" (texte retiré du dépôt : recadrages, miroirs, petites rotations, changements de couleur,
flou, bruit). Mesure, pour l'origine et le meilleur modèle Codex, ce que deviennent pertes et prédictions par type."""
import json, math, sys, torch, torch.nn.functional as F
from pathlib import Path
H = Path.home() / "t3"; sys.path.insert(0, str(H))
import task_template as tt
tt.DEVICE = "cpu"; dev = "cuda"
from t3_perturb_lib import crop, blur, rot
g = torch.Generator().manual_seed(1)
def U(a, b, n): return torch.empty(n).uniform_(a, b, generator=g)
def rcrop(x, lo):                      # recadrage aléatoire redimensionné, échelle [lo,1], ratio [3/4,4/3]
    out = []
    for i in range(len(x)):
        s = float(U(lo, 1, 1)); r = math.exp(float(U(math.log(3 / 4), math.log(4 / 3), 1)))
        w = min(224, int(round(224 * math.sqrt(s * r)))); h = min(224, int(round(224 * math.sqrt(s / r))))
        y0 = int(torch.randint(0, 224 - h + 1, (1,), generator=g)); x0 = int(torch.randint(0, 224 - w + 1, (1,), generator=g))
        out.append(F.interpolate(x[i:i + 1, :, y0:y0 + h, x0:x0 + w], size=224, mode="bilinear", align_corners=False))
    return torch.cat(out)
def rflip(x):
    m = (torch.rand(len(x), generator=g) < 0.5).to(x.device)[:, None, None, None]; return torch.where(m, x.flip(-1), x)
def rrot(x, d):
    t = U(-d, d, len(x)).to(x.device) * math.pi / 180
    A = torch.stack([torch.stack([t.cos(), -t.sin(), torch.zeros_like(t)], 1), torch.stack([t.sin(), t.cos(), torch.zeros_like(t)], 1)], 1)
    return F.grid_sample(x, F.affine_grid(A, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
def colour(x, s):
    n = len(x); d = x.device
    b = (1 + U(-s, s, n)).to(d)[:, None, None, None]; c = (1 + U(-s, s, n)).to(d)[:, None, None, None]; sa = (1 + U(-s, s, n)).to(d)[:, None, None, None]
    x = x * b; m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * c + m
    gr = x.mean(1, keepdim=True); x = (x - gr) * sa + gr
    sh = (U(-s, s, n * 3).view(n, 3, 1, 1) * 0.1).to(d); return (x + sh).clamp(0, 1)
def rblur(x, lo, hi):
    return torch.cat([blur(x[i:i + 1], float(U(lo, hi, 1))) for i in range(len(x))])
def rnoise(x, s): return (x + s * torch.randn(x.shape, device=x.device)).clamp(0, 1)
A = {
    "clean": lambda x: x,
    "crop+flip": lambda x: rflip(rcrop(x, 0.6)),
    "rot10": lambda x: rrot(x, 10), "colour0.2": lambda x: colour(x, 0.2), "colour0.4": lambda x: colour(x, 0.4),
    "blur0.5-1.5": lambda x: rblur(x, 0.5, 1.5), "blur1-2.5": lambda x: rblur(x, 1.0, 2.5), "noise0.03": lambda x: rnoise(x, 0.03), "noise0.08": lambda x: rnoise(x, 0.08),
    "ALL-doux": lambda x: rnoise(rblur(colour(rrot(rflip(rcrop(x, 0.7)), 7), 0.2), 0.3, 1.0), 0.02),
    "ALL-moyen": lambda x: rnoise(rblur(colour(rrot(rflip(rcrop(x, 0.5)), 12), 0.3), 0.5, 1.5), 0.04),
    "ALL-fort": lambda x: rnoise(rblur(colour(rrot(rflip(rcrop(x, 0.35)), 15), 0.4), 0.5, 2.0), 0.06),
}
@torch.no_grad()
def run(m, X, f, bs=100, seed=0):
    g.manual_seed(seed); torch.manual_seed(seed); out = []
    for i in range(0, len(X), bs):
        x = f(((X[i:i + bs].to(dev) + 1) / 2)).clamp(0, 1) * 2 - 1
        with torch.autocast("cuda", dtype=torch.float16): out.append(m(x).float().cpu())
    return torch.cat(out)
X = tt.preprocess(tt.load_forget_images())
models = {"orig": "model.pt"}
for a in sys.argv[1:]: models[Path(a).stem[:28]] = a
m = tt.build_model().to(dev).eval(); res = {}
for name, path in models.items():
    m.load_state_dict(torch.load(H / path, map_location=dev))
    Lc = run(m, X, A["clean"]); y = Lc.argmax(1) if name == "orig" else y; pc = F.softmax(Lc / 2, 1)
    for k, f in A.items():
        L = run(m, X, f, seed=3); ce = F.cross_entropy(L, y, reduction="none"); p = F.softmax(L / 2, 1)
        r = dict(model=name, alt=k, acc=(L.argmax(1) == y).float().mean().item(), ce_mean=ce.mean().item(), ce_std=ce.std().item(),
                 same_top1_as_clean=(L.argmax(1) == Lc.argmax(1)).float().mean().item(),
                 kl_clean_alt=(pc * (pc.clamp_min(1e-9).log() - p.clamp_min(1e-9).log())).sum(1).mean().item())
        print(json.dumps(r), flush=True)
print("DONE")
