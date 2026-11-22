#!/usr/bin/env python3
"""Compact view of a diagnosis.json (one line per endpoint). usage: show.py OUT/diagnosis.json [TRUTH.csv]"""
import csv, json, sys

d = json.load(open(sys.argv[1]))
kind = {}
if len(sys.argv) > 2:
    for r in csv.DictReader(open(sys.argv[2])): kind[r['endpoint']] = r.get('kind') or r['model']
rows_used = d.get('thresholds', {}).get('gfp_rows', 'all')
for fl, F in d['families'].items():
    print(f"== {fl} {F['base']} T0 {F['T0']} date {F.get('template_date')}")
    print('endpoint  nL  tok    T     z0  disp  sd  stop  trunc | prb_own sh_T0  altLLR(prb) for/ag zplain | rep_ex lcp  k | fp8: z0   z1  frac llr@T0 | gfp self z  self_p  best (along p t dose) | proxy self cf/z  pooled cf/z | decision')
    for e, v in sorted(F['endpoints'].items()):
        m = v['measures']; c = m['cal'] or {}; p = m['plain']; r = m['rep']; f8 = m.get('fp8', {})
        fa = f8.get('all') or {}; fo = f8.get('reset') or {}
        sg = v.get('adapter_self'); pg = v.get('adapter_pooled'); fg = v.get('fingerprint')
        gtxt = '        -                              '
        if fg:
            o, bp = min(fg[rows_used]['pairs'].items(), key=lambda kv: kv[1]['p'] if kv[1]['p'] is not None else 1.0)
            gtxt = f"{fg['all']['self']['z']:6.2f} {fg[rows_used]['self'].get('p', 1):.4f}  {o[-2:]} {bp['p'] if bp['p'] is not None else float('nan'):7.4f} {bp['t'] if bp['t'] is not None else float('nan'):5.1f} {bp['ratio'] if bp['ratio'] is not None else float('nan'):5.2f}"
        print(f"{e:7s} {c.get('n_ans', 0):3d} {c.get('tokens', 0):5d} {c.get('T', 0):5.3f} {c.get('z0', 0):6.1f} {c.get('disp', 0):5.1f} {c.get('ans_sd', 0):4.1f} {c.get('z_stop', 0):5.1f} {(m.get('trunc') or {}).get('z', 0):6.1f} | "
              f"{p['probe_fz_own']:6.1f} {p['short_fz_T0']:6.1f} {p.get('alt_llr', 0):7.1f}({p.get('alt_llr_probe', 0):6.1f}) {p.get('alt_n_for', 0):2d}/{p.get('alt_n_against', 0):2d} {p['z_plain_own']:7.1f} | "
              f"{r.get('excess', 0):6.1f} {r['max_lcp']:3d} {r['max_k']:2d} | {fo.get('z0', 0):6.1f} {fo.get('z1', 0):5.1f} {fo.get('frac', 0):5.2f} {fa.get('llr_per_ktok', 0):6.2f} | {gtxt} | "
              + (f"{sg['cf_gain_per_answer']:5.2f}/{sg['cf_z']:5.1f}" if sg else '     -     ') + ' '
              + (f"{pg['cf_gain_per_answer']:5.2f}/{pg['cf_z']:5.1f}" if pg else '     -     ')
              + f" | {v['decision']['kind']:5s} {kind.get(e, '')}")
    for k, x in sorted((F['adapter'].get('cross') or {}).items()):
        print(f"   cross {k}: dose {x['s_hat']} T {x['T']} ratio {x.get('ratio')} cf {x['cf_gain_per_answer']} z {x['cf_z']}")
if 'score' in d: print('SCORE', d['score'])
