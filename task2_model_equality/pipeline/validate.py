#!/usr/bin/env python3
"""Validation of me_pipeline.py on endpoints with a published truth.

usage: validate.py --config CONFIG.json --answers A.jsonl [...] --truth TRUTH.csv --out OUTDIR --cache CACHE
                   [--subsamples 20] [--budget short=16,repeat=8,long=35] [--skip-full] [--first-seed 0]

1. full run on every answer                       -> OUTDIR/full
2. N random subsamples at the scored budget       -> OUTDIR/sub_XX (16 plain short answers with the extraction prompts always kept,
                                                     8 answers of identical requests, 35 reset-prefixed long answers per endpoint)
Writes OUTDIR/validation.json and OUTDIR/validation.md (MCC of every run, error counts per endpoint kind).
"""
import argparse, collections, csv, json, os, random, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import me_pipeline as mp

def subsample(rows, cfg, budget, seed):
    """Scored budget per endpoint: `short` plain answers (one per extraction prompt first, then other plain prompts), `repeat` answers of
    identical reset-prefixed requests, `long` reset-prefixed answers of distinct prompts. Like the real collection, every endpoint gets
    the same prompts: one random order of the prompts is drawn per subsample and each endpoint takes the first ones it has."""
    rnd = random.Random(seed); out = []
    reset = lambda r: any(mk in r['prompt'] for mk in cfg['reset_markers'])
    probe = lambda r: any(r['id'].startswith(p) for p in cfg['probe_prefixes'])
    prompts = sorted({(r['prompt'], r['m']) for r in rows}); rnd.shuffle(prompts)
    rank = {k: i for i, k in enumerate(prompts)}
    by = collections.defaultdict(list)
    for r in rows: by[r['endpoint']].append(r)
    for e in sorted(by):
        rs = list(by[e]); rnd.shuffle(rs)
        groups = collections.defaultdict(list)
        for r in rs: groups[(r['prompt'], r['m'])].append(r)
        order = sorted(groups, key=lambda k: rank[k])
        plain = [k for k in order if not reset(groups[k][0])]
        short = [groups[k][0] for k in plain if probe(groups[k][0])] + [groups[k][0] for k in plain if not probe(groups[k][0])]
        short = short[:budget['short']]
        rep = []
        rk = sorted((k for k in order if reset(groups[k][0]) and len(groups[k]) >= 3), key=lambda k: -min(len(groups[k]), budget['repeat']))
        for k in rk:                                      # largest groups of identical requests first
            room = budget['repeat'] - len(rep)
            if room < 2: break
            rep += groups[k][:room]
        used = {(r['prompt'], r['m']) for r in rep}
        longs = [groups[k][0] for k in order if reset(groups[k][0]) and k not in used][:budget['long']]
        out += short + rep + longs
    return [dict(r) for r in out]

def kinds(truth_csv):
    k = {}
    for r in csv.DictReader(open(truth_csv)): k[r['endpoint']] = r.get('kind') or r['model']
    return k

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True); ap.add_argument('--answers', nargs='+', required=True); ap.add_argument('--truth', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--subsamples', type=int, default=20); ap.add_argument('--first-seed', type=int, default=0)
    ap.add_argument('--budget', default='short=16,repeat=8,long=35'); ap.add_argument('--skip-full', action='store_true')
    ap.add_argument('--no-lora', action='store_true'); ap.add_argument('--no-cross', action='store_true'); ap.add_argument('--proxy', action='store_true')
    ap.add_argument('--redecide', action='store_true', help='reuse the state.pkl of every run (measurements) and only redo the decisions')
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    ov = {}
    for s in a.set:
        k, v = s.split('='); ov[k] = v if isinstance(mp.THR.get(k), str) else type(mp.THR[k])(float(v)) if k in mp.THR else float(v)
    cfg = mp.load_config(a.config, ov)
    budget = {k: int(v) for k, v in (x.split('=') for x in a.budget.split(','))}
    rows = mp.load_rows(a.answers, set(cfg['endpoints']) if cfg.get('endpoints') else None)
    truth_eps = {r['endpoint'] for r in csv.DictReader(open(a.truth))}
    rows = [r for r in rows if r['endpoint'] in truth_eps]
    kd = kinds(a.truth); os.makedirs(a.out, exist_ok=True)
    res_p = os.path.join(a.out, 'validation.json')
    res = json.load(open(res_p)) if os.path.exists(res_p) and not a.redecide else dict(full=None, subs={})
    if a.redecide and os.path.exists(res_p): res['full'] = json.load(open(res_p)).get('full')
    truth_model = {r['endpoint']: (r['family'], r['model']) for r in csv.DictReader(open(a.truth))}
    base_label = collections.Counter(truth_model.values()).most_common(1)[0][0]
    def record(diag, part):
        wrong = {}
        pred_base = collections.Counter(v['model'] for v in part.values()).most_common(1)[0][0]
        for e, v in part.items():
            t_base = truth_model[e] == base_label; p_base = v['model'] == pred_base
            if t_base != p_base: wrong[e] = ('called trap' if t_base else 'called base') + f' ({kd[e]})'
        return dict(score=diag['score'], wrong=wrong, partition={e: v['model'] for e, v in part.items()})
    if not a.skip_full:
        part, diag = mp.run(cfg, [dict(r) for r in rows], os.path.join(a.out, 'full'), a.cache, a.truth, a.no_lora, a.no_cross, a.redecide, a.proxy)
        res['full'] = record(diag, part); json.dump(res, open(res_p, 'w'), indent=1)
    for s in range(a.first_seed, a.first_seed + a.subsamples):
        sub = subsample(rows, cfg, budget, s)
        part, diag = mp.run(cfg, sub, os.path.join(a.out, f'sub_{s:02d}'), a.cache, a.truth, a.no_lora, a.no_cross, a.redecide, a.proxy)
        res['subs'][str(s)] = record(diag, part); json.dump(res, open(res_p, 'w'), indent=1)
        mp.log(f'SUBSAMPLE {s}: model MCC {diag["score"]["mcc_model"]} wrong {res["subs"][str(s)]["wrong"]}')
    md = ['# Validation', '']
    if res.get('full'):
        md += [f"Full data: family MCC {res['full']['score']['mcc_family']}, model MCC {res['full']['score']['mcc_model']}, score {res['full']['score']['score']}; "
               f"endpoints on the wrong side: {res['full']['wrong'] or 'none'}", '']
    if res['subs']:
        ms = [v['score']['mcc_model'] for v in res['subs'].values()]
        md += [f"Subsamples at the budget {a.budget}: {len(ms)} runs, model MCC mean {statistics.mean(ms):.3f}, median {statistics.median(ms):.3f}, "
               f"min {min(ms):.3f}, max {max(ms):.3f}, runs at 1.000: {sum(m >= 0.9999 for m in ms)}", '',
               'Model MCC of each run: ' + ' '.join(f'{m:.2f}' for m in ms), '']
        err = collections.Counter()
        for v in res['subs'].values():
            for e, w in v['wrong'].items(): err[f'{e} {w}'] += 1
        md += ['Errors per endpoint over the subsamples (count of runs):', ''] + [f'- {k}: {n}' for k, n in err.most_common()] + ['']
    open(os.path.join(a.out, 'validation.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))

if __name__ == '__main__':
    main()
