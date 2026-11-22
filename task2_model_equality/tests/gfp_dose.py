"""Dose table from a gradfp.py result: projection of every endpoint on the direction of a reference (strong) endpoint.
dose(x) = pair effect(x, ref) / self effect(ref). 0 = base model, 1 = same strength as the reference, in between = weaker dose of the same adapter.
The 80% interval comes from a bootstrap over prompt folds.
usage: gfp_dose.py gfp_FAM.json ep-strong[,ep-strong2]"""
import json, sys
import numpy as np

d = json.load(open(sys.argv[1])); eps = d['eps']; K = d['folds']; G = np.array(d['gram']); nt = np.array(d['ntok'])
off = ~np.eye(K, dtype=bool)
def eff(a, b, wa=None, wb=None):
    M = G[a * K:(a + 1) * K, b * K:(b + 1) * K]; W = np.outer(nt[a * K:(a + 1) * K], nt[b * K:(b + 1) * K])
    if wa is not None:
        w = np.outer(wa, wb); M = M * w; W = W * w
    return M[off].sum() / max(W[off].sum(), 1)
rng = np.random.default_rng(0)
for s in sys.argv[2].split(','):
    si = eps.index(s); base = eff(si, si)
    print(f'reference {s}: self effect x1e6 {base * 1e6:.0f}')
    for a, e in enumerate(eps):
        if a == si: continue
        v = eff(a, si) / base; bs = []
        for _ in range(400):
            wa = rng.multinomial(K, [1 / K] * K).astype(float); wb = rng.multinomial(K, [1 / K] * K).astype(float)
            den = eff(si, si, wb, wb)
            if den > 0: bs.append(eff(a, si, wa, wb) / den)
        bs.sort(); lo, hi = bs[int(.1 * len(bs))], bs[int(.9 * len(bs))]
        T = d['That'][e]
        print(f'  {e[-2:]}: dose {v:+.2f}  (80% {lo:+.2f} .. {hi:+.2f})  T {T:.2f}')
