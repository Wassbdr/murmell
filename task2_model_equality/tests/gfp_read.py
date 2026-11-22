"""Read a gradfp.py result: self test, pair test, direction cosine and dose ratio.
usage: gfp_read.py gfp_FAM.json [--ref ep-a,ep-b]   (--ref: endpoints taken as the base model, their mean gradient per token is removed
from every endpoint before the tests: cancels what the replica and the reset block do not reproduce exactly)"""
import json, sys, math
import numpy as np

d = json.load(open(sys.argv[1])); eps = d['eps']; K = d['folds']; G = np.array(d['gram']); nt = np.array(d['ntok'])
E = len(eps)
ref = []
if '--ref' in sys.argv: ref = [e for e in sys.argv[sys.argv.index('--ref') + 1].split(',') if e in eps]
if ref:
    # centred sums: S'_ak = S_ak - n_ak * mu, mu = sum of reference sums / reference tokens. A reference endpoint is centred leave-one-out.
    def centre(excl):
        idx = [eps.index(r) * K + k for r in ref if r != excl for k in range(K)]
        w = np.zeros(E * K); w[idx] = 1.0 / nt[idx].sum()
        return w
    A = np.eye(E * K)
    for a, e in enumerate(eps):
        w = centre(e)
        for k in range(K): A[a * K + k] -= nt[a * K + k] * w
    G = A @ G @ A.T

def stat(a, b):
    M = G[a * K:(a + 1) * K, b * K:(b + 1) * K]; W = np.outer(nt[a * K:(a + 1) * K], nt[b * K:(b + 1) * K])
    mask = ~np.eye(K, dtype=bool) if a != b else np.triu(np.ones((K, K), dtype=bool), 1)
    v = M[mask]; w = W[mask]
    return v.sum() / max(math.sqrt((v ** 2).sum()), 1e-30), v.sum() / max(w.sum(), 1)

Z = np.zeros((E, E)); F = np.zeros((E, E))
for a in range(E):
    for b in range(E): Z[a, b], F[a, b] = stat(a, b)
print('centred on', ref if ref else 'nothing')
print('self test: endpoint | fitted T | answers tokens | z | effect x1e6')
for a, e in enumerate(eps):
    print(f"  {e[-2:]} | T {d['That'][e]:.2f} | {d['n'][e]:3d} {int(nt[a * K:(a + 1) * K].sum()):6d} | z {Z[a, a]:+6.2f} | {F[a, a] * 1e6:+9.3f}")
print('pair z:')
print('      ' + ' '.join(f'{e[-2:]:>6s}' for e in eps))
for a, e in enumerate(eps): print(f'  {e[-2:]}  ' + ' '.join(f'{Z[a, b]:+6.1f}' if a != b else '     .' for b in range(E)))
print('direction cosine = pair effect / sqrt(self effect a x self effect b), shown when both self effects are positive:')
print('      ' + ' '.join(f'{e[-2:]:>6s}' for e in eps))
for a, e in enumerate(eps):
    row = []
    for b in range(E):
        if a == b: row.append('     .')
        elif F[a, a] > 0 and F[b, b] > 0: row.append(f'{F[a, b] / math.sqrt(F[a, a] * F[b, b]):+6.2f}')
        else: row.append('     -')
    print(f'  {e[-2:]}  ' + ' '.join(row))
