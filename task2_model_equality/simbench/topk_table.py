"""topk_table.py - the question "can a base endpoint sampled with T = 0.9, top_k = 50 align on a faint LoRA direction?".

    python topk_table.py fam_topk [--ref-dose 0.5] [--folds 24]

For every family of fam_topk/: each endpoint against the reference (the undisguised LoRA endpoint of the dose given, default = the
strongest one), with the gradient computed under the truncated model for the top-k endpoints (gfp_topk.json, option TOPK of gradfp.py)
and under the plain model (gfp_plain.json). Columns: fitted T, self z, self sign-flip p, slope on the reference, cosine, calibrated p."""
import argparse, csv, itertools, json, math, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--ref-dose', type=float, default=None); ap.add_argument('--folds', type=int, default=24); ap.add_argument('--R', type=int, default=50000)
a = ap.parse_args(); rng = np.random.default_rng(0)

def read(p, folds):
    d = json.load(open(p)); eps = d['eps']; F0 = d['folds']; E = len(eps); G = np.array(d['gram']); nt = np.array(d['ntok']); F = folds if F0 % folds == 0 else F0
    if F != F0:
        A = np.zeros((E * F, E * F0))
        for e in range(E):
            for k in range(F0): A[e * F + k % F, e * F0 + k] = 1
        G = A @ G @ A.T; nt = A @ nt
    G = G.reshape(E, F, E, F); n = nt.reshape(E, F); off = ~np.eye(F, dtype=bool); up = np.triu(np.ones((F, F), bool), 1)
    signs = rng.choice([-1.0, 1.0], size=(a.R, F)) if F > 13 else np.array(list(itertools.product([-1.0, 1.0], repeat=F)))
    ix = {e: i for i, e in enumerate(eps)}
    def stat(x, y):
        M = G[ix[x], :, ix[y], :]; mask = up if x == y else off; v = M[mask]; w = np.outer(n[ix[x]], n[ix[y]])[mask]
        return float(v.sum() / max(1e-30, math.sqrt((v ** 2).sum()))), float(v.sum() / max(1.0, w.sum()))
    def along(x, y):
        r = (G[ix[x], :, ix[y], :] * off).sum(1); ok = n[ix[x]] > 0; r = r[ok]
        return float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)) + 1e-30)), float((1 + ((signs[:, ok] @ r) >= r.sum() - 1e-12).sum()) / (1 + len(signs)))
    def pself(x):
        M = G[ix[x], :, ix[x], :] * off; null = np.einsum('rk,kl,rl->r', signs, M, signs) / 2
        return float((1 + (null >= M.sum() / 2 - 1e-12).sum()) / (1 + len(null)))
    return d, stat, along, pself

for name in sorted(os.listdir(a.root)):
    D = os.path.join(a.root, name)
    truth = {r['endpoint']: r for r in csv.DictReader(open(os.path.join(D, 'truth.csv')))}
    for tag, label in (('gfp_topk.json', 'gradient under the TRUNCATED model for the top-k endpoints (TOPK option)'), ('gfp_plain.json', 'gradient under the plain model for every endpoint')):
        p = os.path.join(D, tag)
        if not os.path.exists(p): continue
        d, stat, along, pself = read(p, a.folds)
        plain = [e for e in d['eps'] if truth[e]['model'].startswith('lora') and '+' not in truth[e]['kind']]
        dose = lambda e: float(truth[e]['model'][5:])
        ref = max(plain, key=dose) if a.ref_dose is None else [e for e in plain if abs(dose(e) - a.ref_dose) < 1e-9][0]
        S = {e: stat(e, e) for e in d['eps']}
        print('\n### %s (%s, %s answers per endpoint, %d folds), %s' % (name, d['model'].split('/')[-1], '/'.join(sorted({str(v) for v in d['n'].values()})), d['folds'], label))
        print('Reference: %s (%s), fitted T %.3f, self z %+.1f, self p %.5f, self effect x1e6 %.0f\n' % (ref, truth[ref]['kind'], d['That'][ref], S[ref][0], pself(ref), S[ref][1] * 1e6))
        print('| endpoint | kind | fitted T | self z | self p | self effect x1e6 | slope on ref | cos | t along ref | calibrated p |'); print('|---|---|---|---|---|---|---|---|---|---|')
        for e in sorted(d['eps'], key=lambda e: (truth[e]['model'] != 'base', truth[e]['kind'], e)):
            if e == ref: continue
            z, ef = stat(e, ref); t, pv = along(e, ref); cos = ef / math.sqrt(S[e][1] * S[ref][1]) if S[e][1] > 0 and S[ref][1] > 0 else float('nan')
            print('| %s | %s | %.3f | %+.1f | %.4f | %+.0f | %+.2f | %s | %+.1f | %.5f |' % (e, truth[e]['kind'], d['That'][e], S[e][0], pself(e), S[e][1] * 1e6, ef / S[ref][1] if S[ref][1] > 0 else float('nan'),
                  ('%.2f' % cos) if cos == cos else '-', t, pv))
