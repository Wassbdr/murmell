"""simgrad.py - the gradient fingerprint (hk/mine/gradfp.py) on the simulated families, compared with truth.

    python simgrad.py fam [--mode long] [--folds 12] [--only s64,bank] [--out grad_tables_long.md]

Everything is recomputed from the Gram matrix that gradfp.py stores (fold sums of the per-answer gradients). A run made with
FOLDS = 24 is first aggregated to 12 folds (hash % 24 % 12 = hash % 12), so the numbers are the ones gradfp.py prints with FOLDS=12.

Statistics, for endpoints a and b (M = Gram block of a's folds against b's folds, same-fold terms excluded):
  z(a, b)      gradfp.py: sum(M) / sqrt(sum(M^2))                       (its null assumes BOTH endpoints have mean-zero gradients)
  eff(a, b)    sum(M) / sum(n_a n_b)
  cos(a, b)    eff(a, b) / sqrt(eff(a, a) eff(b, b))                    undefined when a self effect is <= 0
  dose(a, b)   sqrt(eff(a, a) / eff(b, b))      slope(a | b) = eff(a, b) / eff(b, b)   (dose of a measured along b)
  p(a | b)     calibrated test of "a carries the direction of b": r_k = g_a,k . (sum of b's other folds); under H0 (a = base model)
               the r_k are independent with mean zero WHATEVER b is. One-sided sign-flip p-value over a's folds, and Student t of r.
  p_self(a)    sign-flip p-value of the self statistic (signs of a's fold sums).
The rule under test (lead): reference b = strongest undisguised endpoint (largest self effect among endpoints that keep the default
identity); a is "a dose of the LoRA" when z(a, b) >= Z and cos(a, b) >= 0.6, otherwise base."""
import argparse, collections, csv, itertools, json, math, os, re
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--mode', default='long'); ap.add_argument('--out', default=None)
ap.add_argument('--folds', type=int, default=12); ap.add_argument('--only', default=''); ap.add_argument('--R', type=int, default=20000)
ap.add_argument('--flag', default='oracle', help="which endpoints count as prompt-disguised: 'oracle' (truth: sys or persona, i.e. a perfect extraction test) or 'name' (blind NAME-field heuristic)")
a = ap.parse_args()
rng_ = np.random.default_rng(0)


class GF:
    def __init__(self, d, folds):
        self.eps = d['eps']; F0 = d['folds']; E = len(self.eps); G = np.array(d['gram']); nt = np.array(d['ntok'])
        F = folds if (folds and F0 % folds == 0) else F0
        if F != F0:
            A = np.zeros((E * F, E * F0))
            for e in range(E):
                for k in range(F0): A[e * F + k % F, e * F0 + k] = 1
            G = A @ G @ A.T; nt = A @ nt
        self.F = F; self.G = G.reshape(E, F, E, F); self.n = nt.reshape(E, F); self.ix = {e: i for i, e in enumerate(self.eps)}
        self.off = ~np.eye(F, dtype=bool); self.up = np.triu(np.ones((F, F), bool), 1)
        self.signs = rng_.choice([-1.0, 1.0], size=(a.R, F)) if F > 13 else np.array(list(itertools.product([-1.0, 1.0], repeat=F)))
    def M(self, x, y): return self.G[self.ix[x], :, self.ix[y], :]
    def stat(self, x, y, drop=None):
        M = self.M(x, y); w = np.outer(self.n[self.ix[x]], self.n[self.ix[y]]); mask = (self.up if x == y else self.off).copy()
        if drop is not None: mask[drop, :] = False; mask[:, drop] = False
        v = M[mask]; return float(v.sum() / max(1e-30, math.sqrt((v ** 2).sum()))), float(v.sum() / max(1.0, w[mask].sum()))
    def eff(self, x, y, drop=None): return self.stat(x, y, drop)[1]
    def along(self, x, y):
        """x tested along the direction of y: Student t of the fold terms and one-sided sign-flip p-value (signs of x's folds)."""
        M = self.M(x, y) * self.off; r = M.sum(1); ok = self.n[self.ix[x]] > 0; r = r[ok]
        t = float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)) + 1e-30)); T = r.sum()
        null = self.signs[:, ok] @ r; p = float((1 + (null >= T - 1e-12).sum()) / (1 + len(null)))
        return t, p
    def along_set(self, x, refs):
        """x tested along the pooled direction of several references (sum of their other-fold sums): same sign-flip test"""
        r = sum((self.M(x, y) * self.off).sum(1) for y in refs); ok = self.n[self.ix[x]] > 0; r = r[ok]
        t = float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)) + 1e-30)); null = self.signs[:, ok] @ r
        return t, float((1 + (null >= r.sum() - 1e-12).sum()) / (1 + len(null)))
    def pself(self, x):
        M = self.M(x, x) * self.off; T = M.sum() / 2; null = np.einsum('rk,kl,rl->r', self.signs, M, self.signs) / 2
        return float((1 + (null >= T - 1e-12).sum()) / (1 + len(null)))
    def dose_ci(self, x, ref, B=400, q=(0.1, 0.9)):
        """gfp_dose.py: dose = eff(x, ref) / eff(ref, ref), interval by bootstrap of the folds (independent weights for x and ref)"""
        i, j = self.ix[x], self.ix[ref]; Mx = self.G[i, :, j, :]; Mr = self.G[j, :, j, :]; Wx = np.outer(self.n[i], self.n[j]); Wr = np.outer(self.n[j], self.n[j]); off = self.off
        base = Mr[off].sum() / max(Wr[off].sum(), 1); v = (Mx[off].sum() / max(Wx[off].sum(), 1)) / base; bs = []
        for _ in range(B):
            wa = rng_.multinomial(self.F, [1 / self.F] * self.F).astype(float); wb = rng_.multinomial(self.F, [1 / self.F] * self.F).astype(float)
            wr = np.outer(wb, wb); den = (Mr * wr)[off].sum() / max((Wr * wr)[off].sum(), 1)
            if den > 0:
                w = np.outer(wa, wb); bs.append(((Mx * w)[off].sum() / max((Wx * w)[off].sum(), 1)) / den)
        bs.sort(); return float(v), float(bs[int(q[0] * len(bs))]), float(bs[int(q[1] * len(bs))])
    def dose_diff(self, x, y, ref):
        """paired test of equal dose along ref: per-fold projections of x minus those of y, two-sided sign-flip p-value and Student t"""
        i, k, j = self.ix[x], self.ix[y], self.ix[ref]; ok = (self.n[i] > 0) & (self.n[k] > 0)
        rx = (self.G[i, :, j, :] * self.off).sum(1)[ok] / self.n[i][ok]; ry = (self.G[k, :, j, :] * self.off).sum(1)[ok] / self.n[k][ok]; dlt = rx - ry
        t = float(dlt.mean() / (dlt.std(ddof=1) / math.sqrt(len(dlt)) + 1e-30)); null = np.abs(self.signs[:, ok] @ dlt)
        return t, float((1 + (null >= abs(dlt.sum()) - 1e-18).sum()) / (1 + len(null)))
    def jack(self, fn):
        v = np.array([fn(j) for j in range(self.F)]); v = v[np.isfinite(v)]
        return float(math.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum())) if len(v) > 2 else float('nan')


def prompt_flags(p):
    """blind stand-in for the extraction test: NAME field of the ROLE / TONE / NAME line, against the majority of the family"""
    nm = {}
    for l in open(p):
        r = json.loads(l)
        if r['id'] == 'per-extract-summary':
            m = re.search(r'NAME\s*=\s*([^;\n]*)', r['response'] or ''); nm[r['endpoint']] = re.sub(r'[^a-z0-9]+', ' ', m.group(1).lower()).strip()[:20] if m else '?'
    mode = collections.Counter(nm.values()).most_common(1)[0][0]
    return {e: int(v != mode) for e, v in nm.items()}


def sq(x): return math.sqrt(x) if x > 0 else float('nan')
def fmt(v, f='%+.1f'): return '-' if v != v else f % v
def rng(v, f='%+.1f'):
    v = sorted(x for x in v if x == x)
    if not v: return '-'
    return f % v[0] if len(v) == 1 else (f + ' .. ' + f + ' (med ' + f + ')') % (v[0], v[-1], v[len(v) // 2])
def korder(k): m = k.split('+')[0]; return (m != 'base', m, k)
def dose_of(model): return float(model[5:]) if model.startswith('lora') else None

only = [x for x in a.only.split(',') if x]
selfk = collections.defaultdict(list); ref_rows = collections.defaultdict(list); pairs = []; PF = []; eq = []; thr_rows = []; fam_truth = {}; fam_lines = []; nullz = collections.defaultdict(list); refs = []
for name in sorted(os.listdir(a.root)):
    p = os.path.join(a.root, name, 'gradfp_%s.json' % a.mode)
    if not os.path.exists(p) or (only and not any(o in name for o in only)): continue
    d = json.load(open(p)); g = GF(d, a.folds); truth = {r['endpoint']: r for r in csv.DictReader(open(os.path.join(a.root, name, 'truth.csv')))}
    flag = prompt_flags(os.path.join(a.root, name, 'answers.jsonl'))
    if a.flag == 'oracle': flag = {e: int('+sys' in truth[e]['kind'] or '+persona' in truth[e]['kind']) for e in truth}
    eps = sorted(g.eps, key=lambda e: (truth[e]['model'] != 'base', truth[e]['kind'], e))
    S = {e: g.stat(e, e) for e in eps}; PS = {e: g.pself(e) for e in eps}
    cand = [e for e in eps if not flag.get(e, 0) and S[e][1] > 0 and '+wm' not in truth[e]['kind']]   # watermark is found by the repeat test
    b = max(cand, key=lambda e: S[e][1]) if cand else None
    refs.append((name, truth[b]['kind'] if b else '-', S[b][0] if b else float('nan'), PS[b] if b else float('nan')))
    ref_list = ([b] + [e for e in eps if e != b and truth[e]['model'] == truth[b]['model'] and any(t in name for t in ('bank', 'dose', 'faint'))]) if b else []
    fam_lines += ['### %s (%s, %s answers per endpoint, rank %d, %d folds read as %d)' % (name, d['model'].split('/')[-1], '/'.join(sorted({str(v) for v in d['n'].values()})), d['rank'], d['folds'], g.F), '',
                  'Reference (strongest undisguised endpoint, chosen blind): %s (%s).' % (b, truth[b]['kind'] if b else '-'), '',
                  '| endpoint | kind | fitted T | prompt-disguised (%s) | self z |' % a.flag + ' self p (sign flip) | self effect x1e6 | pair z with ref | cos with ref | slope on ref | t along ref | p along ref |', '|' + '---|' * 12]
    for e in eps:
        k = truth[e]['kind']; row = dict(fam=name, kind=k, model=truth[e]['model'], z=S[e][0], eff=S[e][1] * 1e6, p=PS[e], flag=flag.get(e, 0), is_ref=e == b)
        for rb in ref_list:   # replicate banks: every copy of the strongest dose is used as the reference in turn
            if e == rb: continue
            z, ef = g.stat(e, rb); t, pv = g.along(e, rb)
            r2 = dict(row); r2.update(zr=z, cos=ef / math.sqrt(S[e][1] * S[rb][1]) if S[e][1] > 0 and S[rb][1] > 0 else float('nan'), slope=ef / S[rb][1], t=t, pr=pv, ref_kind=truth[rb]['kind'], ref_model=truth[rb]['model'])
            ref_rows[k].append(r2)
            if rb == b: row.update(r2)
        selfk[k].append(row)
        fam_lines.append('| %s | %s | %.2f | %d | %+.1f | %.4f | %+.1f | %s | %s | %s | %s | %s |' % (e, k, d['That'][e], flag.get(e, 0), S[e][0], PS[e], S[e][1] * 1e6,
                         fmt(row.get('zr', float('nan'))), fmt(row.get('cos', float('nan')), '%.2f'), fmt(row.get('slope', float('nan')), '%.2f'), fmt(row.get('t', float('nan'))), fmt(row.get('pr', float('nan')), '%.4f')))
    if len(eps) <= 12:
        fam_lines += ['', 'Pair z of gradfp.py (bold = same model in truth):', '', '| | ' + ' | '.join(truth[e]['kind'] for e in eps) + ' |', '|---|' + '---|' * len(eps)]
        for x in eps: fam_lines.append('| %s | ' % truth[x]['kind'] + ' | '.join(('**%+.1f**' if truth[x]['model'] == truth[y]['model'] and x != y else '%+.1f') % g.stat(x, y)[0] for y in eps) + ' |')
    fam_lines.append('')
    if b:   # threshold study: every endpoint against the strongest dose and against the two strongest doses
        cosv = lambda e, r: (g.eff(e, r) / math.sqrt(S[e][1] * S[r][1])) if S[e][1] > 0 and S[r][1] > 0 else float('nan')
        c2 = [e for e in cand if e != b and g.along(e, b)[1] <= 0.005 and cosv(e, b) >= 0.6]
        b2 = max(c2, key=lambda e: S[e][1]) if c2 else None
        for e in eps:
            if e == b: continue
            p1 = g.along(e, b)[1]
            p2, pp = (g.along(e, b2)[1], g.along_set(e, [b, b2])[1]) if (b2 and e != b2) else (p1, p1)
            thr_rows.append(dict(fam=name, e=e, kind=truth[e]['kind'], model=truth[e]['model'], selfp=PS[e], cos=cosv(e, b), p1=p1, p2=p2, pp=pp,
                                 ref_model=truth[b]['model'], ref2_model=truth[b2]['model'] if b2 else '-', struct=len(eps) <= 12))
        fam_truth[name] = dict(models={e: truth[e]['model'] for e in eps}, ref=b, struct=len(eps) <= 12)
    lo_ = [e for e in eps if truth[e]['model'].startswith('lora') and e not in ref_list and '+' not in truth[e]['kind']]
    if ref_list and truth[b]['model'].startswith('lora') and len(lo_) >= 2 and len(set(truth[e]['model'] for e in lo_)) < len(lo_):
        for rb in ref_list:
            ci = {e: g.dose_ci(e, rb) for e in lo_}
            for x, y in itertools.combinations(lo_, 2):
                if truth[x]['model'][:5] != truth[rb]['model'][:5] or truth[y]['model'][:5] != truth[rb]['model'][:5]: continue
                t, pv = g.dose_diff(x, y, rb); dx, dy = dose_of(truth[x]['model']), dose_of(truth[y]['model'])
                eq.append(dict(fam=name, dl=min(dx, dy), dh=max(dx, dy), ref=dose_of(truth[rb]['model']), sx=ci[x][0], sy=ci[y][0], overlap=not (ci[x][2] < ci[y][1] or ci[y][2] < ci[x][1]),
                               width=(ci[x][2] - ci[x][1] + ci[y][2] - ci[y][1]) / 2, t=t, p=pv))
    for x, y in itertools.combinations(eps, 2):
        z, ef = g.stat(x, y); tx, ty = truth[x], truth[y]; ex, ey = S[x][1], S[y][1]
        hi, lo = (x, y) if ex >= ey else (y, x)
        rec = dict(fam=name, a=truth[lo]['kind'], b=truth[hi]['kind'], ma=truth[lo]['model'], mb=truth[hi]['model'], z=z, cos=ef / math.sqrt(ex * ey) if ex > 0 and ey > 0 else float('nan'),
                   dose=sq(S[lo][1] / S[hi][1]) if S[hi][1] > 0 else float('nan'), slope=ef / S[hi][1] if S[hi][1] > 0 else float('nan'), fa=flag.get(lo, 0), fb=flag.get(hi, 0),
                   t=g.along(lo, hi)[0], p=g.along(lo, hi)[1], zs=(S[lo][0], S[hi][0]))
        if rec['ma'] != 'base' and rec['mb'] != 'base' and rec['ma'][:5] == rec['mb'][:5] and rec['ma'].startswith('lora'):
            rec['dose_se'] = g.jack(lambda j: sq(g.eff(lo, lo, j) / g.eff(hi, hi, j)) if g.eff(hi, hi, j) > 0 else float('nan'))
            rec['slope_se'] = g.jack(lambda j: g.eff(lo, hi, j) / g.eff(hi, hi, j) if g.eff(hi, hi, j) > 0 else float('nan'))
        pairs.append(rec)
        if tx['model'] == 'base' and ty['model'] == 'base' and not flag.get(x, 0) and not flag.get(y, 0): nullz['base / base, no prompt disguise'].append(z)

H = ['## Gradient fingerprint (gradfp.py) on the simulated families: %s answers, %d folds, prompt-disguise flag = %s' % (a.mode, a.folds, a.flag), '',
     'Families read: ' + ', '.join('%s (ref %s, self z %+.1f, p %.4f)' % r for r in refs) + '.', '']

# ---- 1. self test by kind
H += ['### Self test by kind', '', '| kind | truth | n | self z | self effect x1e6 | z >= 3 | z >= 4 | sign-flip p <= 0.01 | p <= 0.001 |', '|' + '---|' * 9]
for k in sorted(selfk, key=korder):
    v = selfk[k]; n = len(v)
    H.append('| %s | %s | %d | %s | %s | %.2f | %.2f | %.2f | %.2f |' % (k, 'base' if k.split('+')[0] == 'base' else 'trap', n, rng([s['z'] for s in v]), rng([s['eff'] for s in v], '%+.0f'),
             sum(s['z'] >= 3 for s in v) / n, sum(s['z'] >= 4 for s in v) / n, sum(s['p'] <= 0.01 for s in v) / n, sum(s['p'] <= 0.001 for s in v) / n))

# ---- 2. the rule: pair with the strongest dose
H += ['', '### Rule "pair with the strongest dose": every endpoint against the reference of its family', '',
      'Share of endpoints called "a dose of the LoRA" under each variant of the rule (truth: only lora kinds should be called). cos undefined (self effect <= 0) counts as failing the cosine condition in the first four columns.', '',
      '| kind | truth | n | pair z | cos | slope on ref | t along ref | z >= 3 and cos >= 0.6 | z >= 4 and cos >= 0.6 | z >= 5 and cos >= 0.6 | z >= 3 alone | p <= 0.01 and (cos >= 0.6 or undefined) | p <= 0.001 and (cos >= 0.6 or undefined) | t >= 4 and (cos >= 0.6 or undefined) |', '|' + '---|' * 14]
def share(v, f): return sum(1 for s in v if f(s)) / len(v)
cosok = lambda s: s['cos'] == s['cos'] and s['cos'] >= 0.6
cosok2 = lambda s: s['cos'] != s['cos'] or s['cos'] >= 0.6
for k in sorted(ref_rows, key=korder):
    v = [s for s in ref_rows[k] if s['ref_model'].startswith('lora')]
    if not v: continue
    H.append('| %s | %s | %d | %s | %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (k, 'dose' if k.startswith('lora') else 'not a dose', len(v), rng([s['zr'] for s in v]), rng([s['cos'] for s in v], '%.2f'),
             rng([s['slope'] for s in v], '%.2f'), rng([s['t'] for s in v]), share(v, lambda s: s['zr'] >= 3 and cosok(s)), share(v, lambda s: s['zr'] >= 4 and cosok(s)), share(v, lambda s: s['zr'] >= 5 and cosok(s)),
             share(v, lambda s: s['zr'] >= 3), share(v, lambda s: s['pr'] <= 0.01 and cosok2(s)), share(v, lambda s: s['pr'] <= 0.001 and cosok2(s)), share(v, lambda s: s['t'] >= 4 and cosok2(s))))
allr = [s for v in ref_rows.values() for s in v if s['ref_model'].startswith('lora')]
pos = [s for s in allr if s['kind'].startswith('lora')]; neg = [s for s in allr if not s['kind'].startswith('lora')]
if pos and neg:
    H += ['', 'Totals over %d true doses and %d other endpoints (missed doses / false doses):' % (len(pos), len(neg)), '']
    for label, f in [('z >= 3 and cos >= 0.6', lambda s: s['zr'] >= 3 and cosok(s)), ('z >= 4 and cos >= 0.6', lambda s: s['zr'] >= 4 and cosok(s)), ('z >= 5 and cos >= 0.6', lambda s: s['zr'] >= 5 and cosok(s)),
                     ('z >= 3 and (cos >= 0.6 or undefined)', lambda s: s['zr'] >= 3 and cosok2(s)), ('z >= 4 and (cos >= 0.6 or undefined)', lambda s: s['zr'] >= 4 and cosok2(s)),
                     ('p <= 0.01 and (cos >= 0.6 or undefined)', lambda s: s['pr'] <= 0.01 and cosok2(s)), ('p <= 0.001 and (cos >= 0.6 or undefined)', lambda s: s['pr'] <= 0.001 and cosok2(s)),
                     ('t >= 4 and (cos >= 0.6 or undefined)', lambda s: s['t'] >= 4 and cosok2(s)), ('t >= 5 and (cos >= 0.6 or undefined)', lambda s: s['t'] >= 5 and cosok2(s)),
                     ('RECOMMENDED: p <= 0.005, and cos >= 0.6 only when the own self test is significant (self p <= 0.01)', lambda s: s['pr'] <= 0.005 and (s['p'] > 0.01 or cosok(s))),
                     ('p <= 0.001, and cos >= 0.6 only when the own self test is significant', lambda s: s['pr'] <= 0.001 and (s['p'] > 0.01 or cosok(s))),
                     ('p <= 0.005 and slope >= 0.10, and cos >= 0.6 only when the own self test is significant', lambda s: s['pr'] <= 0.005 and s['slope'] >= 0.10 and (s['p'] > 0.01 or cosok(s)))]:
        fn = sum(1 for s in pos if not f(s)); fp = sum(1 for s in neg if f(s))
        H.append('- %s: missed %d / %d (%s), false %d / %d (%s)' % (label, fn, len(pos), ', '.join('%s x%d' % kv for kv in collections.Counter(s['kind'] for s in pos if not f(s)).most_common()) or 'none',
                 fp, len(neg), ', '.join('%s x%d' % kv for kv in collections.Counter(s['kind'] for s in neg if f(s)).most_common()) or 'none'))

# ---- 2b. p-value thresholds, single reference against the two strongest doses
def fisher(p1, p2): x = p1 * p2; return x * (1 - math.log(x))
RULES = [('p against the strongest dose', lambda r: r['p1']), ('p against the two strongest doses pooled (one direction = sum of both)', lambda r: r['pp']),
         ('Fisher of the two p', lambda r: fisher(r['p1'], r['p2'])), ('max of the two p', lambda r: max(r['p1'], r['p2']))]
GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
def called(r, f, thr): return f(r) <= thr and (r['selfp'] > 0.01 or (r['cos'] == r['cos'] and r['cos'] >= 0.6))
tr = [r for r in thr_rows if r['ref_model'].startswith('lora')]
if tr:
    pos = [r for r in tr if r['model'].startswith('lora')]; neg = [r for r in tr if not r['model'].startswith('lora')]
    H += ['', '### Calibrated p-value thresholds (cos >= 0.6 required only when the own self test is significant, self p <= 0.01)', '',
          '%d true doses and %d other endpoints, families: %s. Second reference found in %d of %d families.' % (len(pos), len(neg), ', '.join(sorted({r['fam'] for r in tr})),
          len({r['fam'] for r in tr if r['ref2_model'] != '-'}), len({r['fam'] for r in tr})), '',
          '| rule | threshold | missed doses | which | false doses | which |', '|---|---|---|---|---|---|']
    for label, f in RULES:
        for thr in GRID:
            fn = [r for r in pos if not called(r, f, thr)]; fp = [r for r in neg if called(r, f, thr)]
            H.append('| %s | %g | %d / %d | %s | %d / %d | %s |' % (label, thr, len(fn), len(pos), ', '.join('%s x%d' % kv for kv in collections.Counter(r['kind'] for r in fn).most_common()) or '-',
                     len(fp), len(neg), ', '.join('%s x%d' % kv for kv in collections.Counter(r['kind'] for r in fp).most_common()) or '-'))
    H += ['', 'Weak doses, p by rule (weakest two doses of each family):', '', '| family | kind | self p | cos | p strongest | p second | p pooled | Fisher |', '|---|---|---|---|---|---|---|---|']
    for r in sorted(pos, key=lambda r: (r['fam'], dose_of(r['model']))):
        H.append('| %s | %s | %.4f | %s | %.4f | %.4f | %.4f | %.5f |' % (r['fam'], r['kind'], r['selfp'], fmt(r['cos'], '%.2f'), r['p1'], r['p2'], r['pp'], fisher(r['p1'], r['p2'])))
    # pair MCC per family (fp8 taken as found by the likelihood-ratio test, every called dose is its own model)
    def mcc_of(fam, f, thr):
        ft = fam_truth[fam]; lab = {}
        rows_ = {r['e']: r for r in tr if r['fam'] == fam}
        for e, m in ft['models'].items():
            if e == ft['ref'] or m in ('fp8', 'fp8a', 'fp8v', 'int8', 'int4') or (e in rows_ and called(rows_[e], f, thr)): lab[e] = 'trap-' + e
            else: lab[e] = 'base'
        tp = fp = fn = tn = 0
        for x, y in itertools.combinations(sorted(lab), 2):
            t, gg = ft['models'][x] == ft['models'][y], lab[x] == lab[y]
            tp += t and gg; fp += (not t) and gg; fn += t and not gg; tn += (not t) and not gg
        d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0
    sf = sorted({r['fam'] for r in tr if r['struct']})
    if sf:
        H += ['', 'Pair MCC of the resulting partition, mean over the %d families of 11 endpoints (%s); 95%% interval by bootstrap of the families; fp8 taken as found by its own test:' % (len(sf), ', '.join(sf)), '',
              '| rule | ' + ' | '.join('p <= %g' % t for t in GRID) + ' |', '|---|' + '---|' * len(GRID)]
        for label, f in RULES:
            cells = []
            for thr in GRID:
                v = np.array([mcc_of(fam, f, thr) for fam in sf]); bs = [v[rng_.integers(0, len(v), len(v))].mean() for _ in range(2000)]
                cells.append('%.3f (%.2f .. %.2f), min %.2f' % (v.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), v.min()))
            H.append('| %s | ' % label + ' | '.join(cells) + ' |')

# ---- 3. calibration of z
H += ['', '### Calibration of the gradfp z (heavy tails)', '', '| pairs | truth | n | z quantiles 50 / 90 / 95 / 99 / max | share z >= 3 | share z >= 4 | share sign-flip p <= 0.01 | share p <= 0.001 |', '|' + '---|' * 8]
def qline(label, tr, v, key='z'):
    if not v: return
    z = np.array([s[key] for s in v]); pv = np.array([s['p'] for s in v])
    H.append('| %s | %s | %d | %+.1f / %+.1f / %+.1f / %+.1f / %+.1f | %.3f | %.3f | %.3f | %.3f |' % (label, tr, len(z), *np.quantile(z, [0.5, 0.9, 0.95, 0.99]), z.max(), (z >= 3).mean(), (z >= 4).mean(), (pv <= 0.01).mean(), (pv <= 0.001).mean()))
plainb = lambda s, side: s['m' + side] == 'base' and not s['f' + side]
qline('base / base, neither prompt-disguised', 'null', [s for s in pairs if plainb(s, 'a') and plainb(s, 'b')])
qline('base (no prompt disguise) along a LoRA dose >= 0.7', 'null', [s for s in pairs if plainb(s, 'a') and (dose_of(s['mb']) or 0) >= 0.7])
for kk in ('base+wm', 'base+T1.0', 'base+T0.85', 'base+T1.0+K50', 'base+T0.9+K50', 'base'):
    qline('%s along a LoRA dose >= 0.7' % kk, 'null', [s for s in pairs if s['a'] == kk and (dose_of(s['mb']) or 0) >= 0.7])
qline('fp8 along a LoRA dose >= 0.7', 'null (different trap)', [s for s in pairs if s['ma'] in ('fp8', 'fp8a', 'fp8v', 'int8', 'int4') and (dose_of(s['mb']) or 0) >= 0.7])
qline('prompt-disguised base along a LoRA dose >= 0.7', 'null', [s for s in pairs if s['ma'] == 'base' and s['fa'] and (dose_of(s['mb']) or 0) >= 0.7 and not s['fb']])
qline('base or fp8 along a prompt-disguised base', 'null', [s for s in pairs if plainb(s, 'a') and s['mb'] == 'base' and s['fb']] + [s for s in pairs if s['ma'] in ('fp8', 'fp8a', 'fp8v', 'int8', 'int4') and s['mb'] == 'base' and s['fb']])
qline('two prompt-disguised base endpoints', 'same model, shared prompt direction', [s for s in pairs if s['ma'] == 'base' and s['mb'] == 'base' and s['fa'] and s['fb']])
qline('LoRA dose along a stronger dose of the same adapter', 'alternative', [s for s in pairs if dose_of(s['ma']) and dose_of(s['mb']) and s['ma'][:5] == s['mb'][:5] and s['ma'] != s['mb']])

# ---- 4. same model against neighbour dose
H += ['', '### Same model against another dose of the same adapter (effect ratio)', '',
      'dose = sqrt(self effect of the weaker / self effect of the stronger); slope = effect(weak, strong) / effect(strong, strong); se by jackknife over folds.', '',
      '| doses (weak / strong) | truth | n pairs | true ratio | dose ratio | its jackknife se | slope | its jackknife se | pair z | cos | share dose ratio >= 0.80 | share slope >= 0.80 |', '|' + '---|' * 12]
grp = collections.defaultdict(list)
for s in pairs:
    da, db = dose_of(s['ma']), dose_of(s['mb'])
    if da and db and s['ma'][:5] == s['mb'][:5] and not s['fa'] and not s['fb']: grp[tuple(sorted((da, db)))].append(s)
for (lo_, hi_), v in sorted(grp.items(), key=lambda kv: (kv[0][0] != kv[0][1], -kv[0][1], -kv[0][0])):
    H.append('| %g / %g | %s | %d | %.2f | %s | %s | %s | %s | %s | %s | %.2f | %.2f |' % (lo_, hi_, 'same model' if lo_ == hi_ else 'different', len(v), lo_ / hi_, rng([s['dose'] for s in v], '%.2f'), rng([s.get('dose_se', float('nan')) for s in v], '%.2f'),
             rng([s['slope'] for s in v], '%.2f'), rng([s.get('slope_se', float('nan')) for s in v], '%.2f'), rng([s['z'] for s in v]), rng([s['cos'] for s in v], '%.2f'),
             share(v, lambda s: s['dose'] == s['dose'] and s['dose'] >= 0.8), share(v, lambda s: s['slope'] >= 0.8)))
if eq:
    H += ['', '### Dose equality test: two endpoints of the same adapter, same dose or neighbouring doses', '',
          'dose = eff(x, ref) / eff(ref, ref) as in gfp_dose.py, ref = a copy of the strongest dose. "intervals overlap" = the two 80% bootstrap intervals of gfp_dose.py overlap (read as: same dose).',
          '"paired p" = two-sided sign-flip test of the per-fold difference of the projections of x and y on the reference (H0: same model).', '',
          '| doses x / y | truth | true ratio | n (pairs x refs) | dose of x | dose of y | ratio small / large | mean width of the 80% interval | intervals overlap | paired p > 0.10 (called same) | paired p > 0.05 | paired p > 0.01 | ratio >= 0.85 | ratio >= 0.75 |', '|' + '---|' * 14]
    ge = collections.defaultdict(list)
    for r in eq: ge[(r['dl'], r['dh'])].append(r)
    def ratio(r):
        lo2, hi2 = sorted((r['sx'], r['sy'])); return lo2 / hi2 if hi2 > 0 else float('nan')
    for (dl, dh), v in sorted(ge.items(), key=lambda kv: (kv[0][0] != kv[0][1], -kv[0][1], -kv[0][0])):
        xs = [r['sx'] if True else 0 for r in v]
        H.append('| %g / %g | %s | %.2f | %d | %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (dl, dh, 'same model' if dl == dh else 'different', dl / dh, len(v),
                 rng([min(r['sx'], r['sy']) for r in v], '%.2f'), rng([max(r['sx'], r['sy']) for r in v], '%.2f'), rng([ratio(r) for r in v], '%.2f'), sum(r['width'] for r in v) / len(v),
                 share(v, lambda r: r['overlap']), share(v, lambda r: r['p'] > 0.10), share(v, lambda r: r['p'] > 0.05), share(v, lambda r: r['p'] > 0.01), share(v, lambda r: ratio(r) >= 0.85), share(v, lambda r: ratio(r) >= 0.75)))
    same = [r for r in eq if r['dl'] == r['dh']]; diff = [r for r in eq if r['dl'] != r['dh']]
    if same and diff:
        H += ['', 'Totals (same-dose pairs wrongly split / different-dose pairs wrongly merged):', '']
        for label, f in [('80% intervals overlap', lambda r: r['overlap']), ('paired p > 0.10', lambda r: r['p'] > 0.10), ('paired p > 0.05', lambda r: r['p'] > 0.05), ('paired p > 0.01', lambda r: r['p'] > 0.01),
                         ('ratio >= 0.85', lambda r: ratio(r) >= 0.85), ('ratio >= 0.75', lambda r: ratio(r) >= 0.75)]:
            H.append('- %s: split %d / %d, merged %d / %d (%s)' % (label, sum(1 for r in same if not f(r)), len(same), sum(1 for r in diff if f(r)), len(diff),
                     ', '.join('%g/%g x%d' % (k[0], k[1], c) for k, c in collections.Counter((r['dl'], r['dh']) for r in diff if f(r)).most_common()) or 'none'))
tw = [s for s in pairs if s['ma'] == s['mb'] and s['ma'] != 'base' and (s['a'] != s['b'])]
if tw:
    H += ['', 'Same trap served twice, one copy disguised:', '', '| family | weak-effect copy / strong-effect copy | pair z | cos | dose ratio | slope | p along |', '|---|---|---|---|---|---|---|']
    for s in tw: H.append('| %s | %s / %s | %+.1f | %s | %s | %.2f | %.4f |' % (s['fam'], s['a'], s['b'], s['z'], fmt(s['cos'], '%.2f'), fmt(s['dose'], '%.2f'), s['slope'], s['p']))
out = '\n'.join(H + ['', '## Per family', ''] + fam_lines) + '\n'
tag = ('_' + a.only.replace(',', '_')) if a.only else ''
open(a.out or os.path.join(os.path.dirname(os.path.abspath(a.root)), 'grad_tables_%s%s.md' % (a.mode, tag)), 'w').write(out); print(out)
