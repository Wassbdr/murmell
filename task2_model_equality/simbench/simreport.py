"""simreport.py - aggregates every scored family of fam/ into the tables of REPORT.md (written to tables.md).

    python simreport.py fam [--out tables.md]

Tables: MCC per family; kind versus statistic across families; decision by kind; wrong pairs by kind; threshold margins."""
import argparse, collections, csv, itertools, json, math, os, sys

ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--out', default=None); a = ap.parse_args()
THR = dict(rep=8, T=0.95, fp8=2.5, gain=0.25)


def decide(st, eps, train, thr=THR):
    why = {}
    for e in eps:
        if st[e]['rep_pfB'] >= thr['rep']: why[e] = 'base:wm'
        elif st[e]['T'] >= thr['T']: why[e] = 'base:T'
    s = sorted(eps, key=lambda e: -st[e]['fp8'])
    if st[s[0]]['fp8'] - st[s[1]]['fp8'] >= thr['fp8'] and s[0] not in why: why[s[0]] = 'fp8'
    for e in train:
        if e not in why: why[e] = 'lora:train'
    for e in eps:
        if e not in why and st[e].get('gain', 0) >= thr['gain']: why[e] = 'lora:gain'
    for e in eps:
        if e not in why: why[e] = 'base:none'
    return why


def mcc(truth, guess, names):
    tp = fp = fn = tn = 0; wrong = []
    for x, y in itertools.combinations(sorted(names), 2):
        t, g = truth[x] == truth[y], guess[x] == guess[y]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
        if t != g: wrong.append((x, y, 'false merge' if g else 'missed same'))
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (max(0.0, (tp * tn - fp * fn) / d) if d else 0.0), wrong


fams = []
for name in sorted(os.listdir(a.root)):
    p = os.path.join(a.root, name)
    if not os.path.exists(os.path.join(p, 'stats.json')): continue
    S = json.load(open(os.path.join(p, 'stats.json'))); truth = {r['endpoint']: r for r in csv.DictReader(open(os.path.join(p, 'truth.csv')))}
    meta = json.load(open(os.path.join(p, 'meta.json')))
    pipe = None
    pp = os.path.join(p, 'pipeline', 'diagnosis.json')
    if os.path.exists(pp):
        dj = json.load(open(pp))
        if 'score' in dj: pipe = dict(mcc=dj['score']['mcc_model'], pairs=dj['score']['model_pairs'], part=dj['partition'])
    fams.append(dict(name=name, S=S, truth=truth, meta=meta, order=meta.get('order', 0), pipe=pipe))
order = ['qwen15_7p4', 'llama1_7p4', 'qwen15_5p6twice', 'llama1_int8', 'qwen05_7p4', 'smol17_7p4']
fams.sort(key=lambda f: (next((i for i, o in enumerate(order) if f['name'].startswith(o)), 99), f['name']))

L = []
L += ['## MCC per simulated family', '', '| family | base | composition (same + traps) | MCC crude decide.py | MCC pipeline | adapter proxy trained on | fp8 margin (best kind) | wrong pairs |', '|---|---|---|---|---|---|---|---|']
kindstat = collections.defaultdict(list); dec = collections.defaultdict(collections.Counter); wrongk = collections.Counter(); margins = []
for f in fams:
    st = f['S']['endpoints']; tr = f['truth']; eps = sorted(st); train = f['S']['train']
    why = decide(st, eps, train); guess = {e: ('base' if why[e].startswith('base') else 'trap-' + e) for e in eps}
    tm = {e: tr[e]['model'] for e in eps}; m, wrong = mcc(tm, guess, eps)
    nb = sum(tm[e] == 'base' for e in eps)
    fs = sorted(eps, key=lambda e: -st[e]['fp8']); margin = st[fs[0]]['fp8'] - st[fs[1]]['fp8']
    wk = collections.Counter('%s: %s / %s' % (k, *sorted([tr[x]['kind'], tr[y]['kind']])) for x, y, k in wrong)
    for k, v in wk.items(): wrongk[k] += v
    L.append('| %s | %s | %d + %d | **%.3f** | %s | %s | %.1f (%s) | %s |' % (f['name'], f['S']['model'].split('/')[-1], nb, len(eps) - nb, m,
             ('%.3f' % f['pipe']['mcc']) if f['pipe'] else '-', ', '.join(tr[e]['kind'] for e in train), margin, tr[fs[0]]['kind'], '; '.join('%d x %s' % (v, k) for k, v in wk.most_common()) or 'none'))
    for e in eps:
        k = tr[e]['kind']; s = dict(st[e]); s['fam'] = f['name']; s['train'] = e in train; s['fp8_rank'] = fs.index(e) + 1
        s['fp8_gap'] = st[e]['fp8'] - max(st[x]['fp8'] for x in eps if x != e); kindstat[k].append(s); dec[k][why[e]] += 1
    f['mcc'] = m

def rng(vals, fmt='%.2f'):
    vals = sorted(vals)
    if not vals: return '-'
    if len(vals) == 1: return fmt % vals[0]
    return (fmt + ' .. ' + fmt + ' (med ' + fmt + ')') % (vals[0], vals[-1], vals[len(vals) // 2])

def korder(k):
    m = k.split('+')[0]
    return (m != 'base', m, k)

L += ['', '## Kind versus statistic (range over the families where the kind appears; n = number of simulated endpoints)', '',
      '| kind | truth | n | fitted T | z at 0.7 | fp8 ratio /1000 tok | fp8 gap to best other | adapter gain / answer (not trained on) | repeat prefix, digit probe (chars) | repeat prefix, any repeated prompt | identity lost (share) | decided as |', '|' + '---|' * 12]
for k in sorted(kindstat, key=korder):
    v = kindstat[k]; g = [s['gain'] for s in v if 'gain' in s and not s['train']]
    L.append('| %s | %s | %d | %s | %s | %s | %s | %s%s | %s | %s | %.2f | %s |' % (k, k.split('+')[0], len(v), rng([s['T'] for s in v]), rng([s['z07'] for s in v], '%+.1f'),
             rng([s['fp8'] for s in v], '%+.1f'), rng([s['fp8_gap'] for s in v], '%+.1f'), rng(g), (' [trained on %d x]' % sum(s['train'] for s in v)) if any(s['train'] for s in v) else '',
             rng([s['rep_pfB'] for s in v], '%d'), rng([s['rep_any'] for s in v], '%d'), sum(s['prompt_flag'] for s in v) / len(v),
             ', '.join('%s %d' % kv for kv in dec[k].most_common())))
extra = sorted({key for v in kindstat.values() for s in v for key in s if key in ('Awa', 'i8', 'i4g')})
if extra:
    L += ['', '## Other quantized copies (likelihood ratio against the base copy, nats per 1000 tokens, long answers)', '', '| kind | n | fp8 weights (Aw) | ' + ' | '.join({'Awa': 'fp8 weights + activations (Awa)', 'i8': 'int8 weights (i8)', 'i4g': '4-bit weights (i4g)'}[x] for x in extra) + ' |', '|' + '---|' * (3 + len(extra))]
    for k in sorted(kindstat, key=korder):
        v = kindstat[k]
        L.append('| %s | %d | %s | %s |' % (k, len(v), rng([s['fp8'] for s in v], '%+.1f'), ' | '.join(rng([s[x] for s in v if x in s], '%+.1f') for x in extra)))
L += ['', '## Wrong pairs by kind, all families', ''] + ['- %d x %s' % (v, k) for k, v in wrongk.most_common()]

# threshold margins: for each rule, the worst case on each side
def allv(pred, key, skip_train=False):
    return [(s[key], k, s['fam']) for k, v in kindstat.items() for s in v if pred(k) and key in s and not (skip_train and s['train'])]
L += ['', '## Thresholds: worst cases on each side', '']
wm = allv(lambda k: '+wm' in k, 'rep_pfB'); nwm = allv(lambda k: '+wm' not in k, 'rep_pfB')
if wm and nwm: L.append('- Watermark, repeated prefix on the digit probe (rule: >= 8 chars): watermarked endpoints %s; lowest %s. Not watermarked: highest %s.' % (rng([x[0] for x in wm], '%d'), min(wm), max(nwm)))
hot = allv(lambda k: '+T1.0' in k, 'T'); nh = allv(lambda k: '+T1.0' not in k, 'T')
if hot and nh: L.append('- Sampling at T = 1.0 (rule: fitted T >= 0.95): T = 1.0 endpoints %s. All others: highest %s.' % (rng([x[0] for x in hot]), max(nh)))
fq = allv(lambda k: k.split('+')[0] in ('fp8', 'fp8v'), 'fp8_gap'); nq = allv(lambda k: k.split('+')[0] not in ('fp8', 'fp8v'), 'fp8_gap')
if fq and nq: L.append('- fp8 (rule: best ratio beats the runner-up by >= 2.5): gap of the true fp8 endpoints %s; lowest %s. Best gap of any other endpoint: %s.' % (rng([x[0] for x in fq], '%+.1f'), min(fq), max(nq)))
lg = allv(lambda k: k.startswith('lora'), 'gain', True); bg = allv(lambda k: k.split('+')[0] == 'base', 'gain', True); og = allv(lambda k: not k.startswith('lora') and k.split('+')[0] != 'base', 'gain', True)
if lg and bg: L.append('- Fitted adapter (rule: gain >= 0.25 per answer, endpoints not trained on): LoRA endpoints lowest %s, highest %s. Base-model endpoints: highest %s. Other traps: highest %s.' % (min(lg), max(lg), max(bg), max(og) if og else '-'))
out = '\n'.join(L) + '\n'
open(a.out or os.path.join(os.path.dirname(os.path.abspath(a.root)), 'tables.md'), 'w').write(out); print(out)
