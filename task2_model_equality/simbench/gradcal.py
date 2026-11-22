"""gradcal.py - calibrated reading of a gradfp.py output (no truth needed, works on the real families).

    python gradcal.py OUT.json [--ref ep-28] [--folds 12]

Why: the z printed by gradfp.py for a pair (a, b) assumes that BOTH endpoints have mean-zero gradients. As soon as b is a real trap
(or a prompt disguise), the terms g_a,k . g_b,l of the same fold k of a are all multiplied by the same strong direction of b: they
are no longer independent, the true number of terms is the number of folds, not folds x (folds - 1), and z is inflated (up to about
sqrt(folds - 1)). On the simulated families this gives fp8 against a strong LoRA at +4.5 with nothing behind it.
Calibrated test of "a carries the direction of b" (H0: a is the base model, whatever b is):
    r_k = g_a,k . (sum of the folds of b other than k)      independent over k, mean zero under H0
    p(a | b) = one-sided sign-flip p-value over the folds of a (exact for <= 13 folds, smallest value 1 / 2^folds), t = Student t of r.
It is NOT symmetric: p(a | b) tests a. Read the row of the endpoint you doubt, in the column of the trap you trust.
Also printed: cos(a, b) = eff(a, b) / sqrt(eff(a, a) eff(b, b)) and slope(a | b) = eff(a, b) / eff(b, b) (dose of a along b)."""
import argparse, itertools, json, math
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('json'); ap.add_argument('--ref', default=None); ap.add_argument('--folds', type=int, default=None); ap.add_argument('--R', type=int, default=20000)
a = ap.parse_args()
d = json.load(open(a.json)); eps = d['eps']; F0 = d['folds']; E = len(eps); G = np.array(d['gram']); nt = np.array(d['ntok'])
F = a.folds if (a.folds and F0 % a.folds == 0) else F0
if F != F0:
    A = np.zeros((E * F, E * F0))
    for e in range(E):
        for k in range(F0): A[e * F + k % F, e * F0 + k] = 1
    G = A @ G @ A.T; nt = A @ nt
G = G.reshape(E, F, E, F); n = nt.reshape(E, F); off = ~np.eye(F, dtype=bool); up = np.triu(np.ones((F, F), bool), 1)
signs = np.random.default_rng(0).choice([-1.0, 1.0], size=(a.R, F)) if F > 13 else np.array(list(itertools.product([-1.0, 1.0], repeat=F)))

def stat(i, j):
    M = G[i, :, j, :]; mask = up if i == j else off; v = M[mask]; w = np.outer(n[i], n[j])[mask]
    return float(v.sum() / max(1e-30, math.sqrt((v ** 2).sum()))), float(v.sum() / max(1.0, w.sum()))
def along(i, j):
    r = (G[i, :, j, :] * off).sum(1); ok = n[i] > 0; r = r[ok]
    t = float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)) + 1e-30)); null = signs[:, ok] @ r
    return t, float((1 + (null >= r.sum() - 1e-12).sum()) / (1 + len(null)))
def pself(i):
    M = G[i, :, i, :] * off; null = np.einsum('rk,kl,rl->r', signs, M, signs) / 2
    return float((1 + (null >= M.sum() / 2 - 1e-12).sum()) / (1 + len(null)))

S = [stat(i, i) for i in range(E)]
print('folds %d (stored %d), smallest possible p %.5f' % (F, F0, 1 / (1 + len(signs))))
print('self test: endpoint | T | gradfp z | sign-flip p | effect x1e6 | sqrt(effect) relative to the largest')
mx = max(s[1] for s in S)
for i, e in enumerate(eps):
    print('  %s | %.2f | %+6.2f | %.5f | %+10.1f | %s' % (e, d['That'][e], S[i][0], pself(i), S[i][1] * 1e6, ('%.2f' % math.sqrt(S[i][1] / mx)) if S[i][1] > 0 and mx > 0 else '-'))
if a.ref:
    j = eps.index(a.ref)
    print('against the reference %s: endpoint | gradfp z | cos | slope | t | calibrated p' % a.ref)
    for i, e in enumerate(eps):
        if i == j: continue
        z, ef = stat(i, j); t, p = along(i, j)
        cos = ef / math.sqrt(S[i][1] * S[j][1]) if S[i][1] > 0 and S[j][1] > 0 else float('nan')
        print('  %s | %+6.2f | %s | %+.2f | %+6.2f | %.5f' % (e, z, ('%.2f' % cos) if cos == cos else '  - ', ef / S[j][1] if S[j][1] > 0 else float('nan'), t, p))
print('calibrated p(row | column) x 1000 (row = endpoint under test, column = direction):')
print('        ' + ' '.join('%6s' % e[-5:] for e in eps))
for i, e in enumerate(eps):
    print('  %5s ' % e[-5:] + ' '.join(('%6s' % '.') if i == j else '%6.1f' % (1000 * along(i, j)[1]) for j in range(E)))
