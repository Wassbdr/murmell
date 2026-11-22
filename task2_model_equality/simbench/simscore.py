"""simscore.py - the raw statistics of our method on one simulated family, then the decide.py rules, then the MCC against truth.

    python simscore.py fam/qwen15_7p4_s0 [--variants Aw,Awa,i8] [--no-proxy] [--train ep-03,ep-07]

Same computations as the scripts of hk/mine, with the base model as a parameter (they hard-code the three real families):
  calib.py           -> fitted temperature (maximum likelihood under the base copy), z at T = 0.7, top-1 deficit, on the long answers
  quant_llr.py "Aw"  -> likelihood ratio fp8 copy against base copy, nats per 1000 tokens (long answers, and all answers)
  repeat test        -> longest common prefix (characters) between two answers to the SAME prompt (digit probe, and any repeated prompt)
  extraction probes  -> does the endpoint keep the default identity (NAME / ORG fields), compared with the majority of the family
  proxy_lora_fast.py -> fitted-adapter gain per answer (the lead's script itself, NOFOLD=1, trained on the two hottest undisguised endpoints)
  method/decide.py   -> rules 1 to 6, base group = one model, every trap alone
Writes stats.json, decision.csv and score.md in the family folder. Truth is only read at the very end, for the evaluation."""
import argparse, collections, csv, itertools, json, math, os, re, subprocess, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
F8 = torch.float8_e4m3fn
ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--variants', default='Aw'); ap.add_argument('--no-proxy', action='store_true')
ap.add_argument('--train', default=None); ap.add_argument('--base', default=None); ap.add_argument('--eval-only', action='store_true'); ap.add_argument('--redo', action='store_true')
a = ap.parse_args()
D = a.dir; meta = json.load(open(os.path.join(D, 'meta.json'))); MODEL = a.base or meta['base']
rows = [json.loads(l) for l in open(os.path.join(D, 'answers.jsonl'))]
eps = sorted({r['endpoint'] for r in rows})
SGRID = '0,0.1,0.2,0.35,0.5,0.75,1.0'


def lcp(x, y):
    n = 0
    for u, v in zip(x, y):
        if u != v: break
        n += 1
    return n


def mcc_pairs(truth, guess, names):
    tp = fp = fn = tn = 0; wrong = []
    for x, y in itertools.combinations(sorted(names), 2):
        t, g = truth[x] == truth[y], guess[x] == guess[y]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
        if t != g: wrong.append((x, y, 'FALSE MERGE' if g else 'MISSED SAME'))
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (max(0.0, (tp * tn - fp * fn) / d) if d else 0.0), dict(tp=tp, fp=fp, fn=fn, tn=tn), wrong


def decide(st, train, thr=dict(rep=8, T=0.95, fp8=2.5, gain=0.25)):
    """method/decide.py, rules in the same order. Returns {endpoint: (label, reason)}."""
    why = {}
    for e in eps:
        if st[e]['rep_pfB'] >= thr['rep']: why[e] = ('base', 'watermark: repeated prefix %d chars' % st[e]['rep_pfB'])
        elif st[e]['T'] >= thr['T']: why[e] = ('base', 'sampling disguise: T %.2f' % st[e]['T'])
    s = sorted(eps, key=lambda e: -st[e]['fp8'])
    if st[s[0]]['fp8'] - st[s[1]]['fp8'] >= thr['fp8'] and s[0] not in why: why[s[0]] = ('fp8', 'fp8 ratio %+.1f vs %+.1f' % (st[s[0]]['fp8'], st[s[1]]['fp8']))
    for e in train:
        if e not in why: why[e] = ('lora', 'adapter proxy trained on it (T %.2f)' % st[e]['T'])
    for e in eps:
        if e not in why and st[e].get('gain', 0) >= thr['gain']: why[e] = ('lora', 'adapter gain %.2f per answer' % st[e]['gain'])
    for e in eps:
        if e not in why: why[e] = ('base', 'no trap signal (T %.2f, fp8 %+.1f, adapter %.2f)' % (st[e]['T'], st[e]['fp8'], st[e].get('gain', 0)))
    return why


stats_p = os.path.join(D, 'stats.json'); part_p = os.path.join(D, 'stats_partial.json')
if a.eval_only:
    S = json.load(open(stats_p)); st = S['endpoints']; train = S['train']
elif os.path.exists(part_p) and not a.redo:
    st = json.load(open(part_p))['endpoints']; print('partial statistics reused', flush=True)
else:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    EOT = tok.convert_tokens_to_ids('<|eot_id|>') if 'lama' in MODEL else tok.convert_tokens_to_ids('<|im_end|>')
    seqs = []
    for r in rows:
        if not r.get('response'): continue
        text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True)
        ctx = tok(text, add_special_tokens=False).input_ids; m = int(float(r.get('max_tokens') or 64))
        x = tok(r['response'], add_special_tokens=False).input_ids
        if len(x) < m: x = x + [EOT]
        x = x[:m]
        if x: seqs.append((r['endpoint'], r['id'], ctx, x))
    TS = [0.40 + 0.02 * i for i in range(46)]; i7 = min(range(len(TS)), key=lambda i: abs(TS[i] - 0.7))
    acc = {e: dict(n=0, ans=0, top1=0, ll=[0.0] * len(TS), S7=[0.0, 0.0, 0.0, 0.0], per=[]) for e in eps}
    base_lp = []
    def zstats(lg, y, T):
        lp = torch.log_softmax(lg / T, -1); p = lp.exp(); ly = lp.gather(1, y[:, None])[:, 0]
        H = -(p * lp).sum(-1); Vv = (p * lp * lp).sum(-1) - H * H
        return float(ly.sum()), float(H.sum()), float(Vv.sum()), float(p.max(-1).values.sum())
    with torch.no_grad():
        for e, rid, ctx, x in seqs:
            lg = model(torch.tensor([ctx + x]).cuda()).logits[0, len(ctx) - 1:-1].double(); y = torch.tensor(x).cuda()
            base_lp.append(torch.log_softmax(lg / 0.7, -1).gather(1, y[:, None])[:, 0].cpu())
            if not rid.startswith('r256-') or len(x) < 8: continue
            d = acc[e]; d['n'] += len(x); d['ans'] += 1; d['top1'] += int((lg.argmax(-1) == y).sum())
            lf = lg.float(); gy = lf.gather(1, y[:, None])[:, 0]
            for i, T in enumerate(TS): d['ll'][i] += float((gy / T - torch.logsumexp(lf / T, -1)).double().sum())   # log-likelihood at T (calib.py S[i][0])
            s7 = zstats(lg, y, 0.7)
            for k in range(4): d['S7'][k] += s7[k]
            d['per'].append((s7[0] + s7[1]) / math.sqrt(s7[2]))
    st = {}
    for e in eps:
        d = acc[e]; z7 = (d['S7'][0] + d['S7'][1]) / math.sqrt(d['S7'][2])
        ib = max(range(len(TS)), key=lambda i: d['ll'][i])
        exp1 = d['S7'][3] / d['n']; r1 = d['top1'] / d['n']; z1 = (r1 - exp1) / math.sqrt(exp1 * (1 - exp1) / d['n'])
        mu = sum(d['per']) / len(d['per']); sd = (sum((v - mu) ** 2 for v in d['per']) / max(1, len(d['per']) - 1)) ** 0.5
        st[e] = dict(long_answers=d['ans'], long_tokens=d['n'], T=round(TS[ib], 2), z07=round(z7, 2), top1z=round(z1, 2), sd07=round(sd, 2))
    # second pass: dispersion of the per-answer z at the endpoint's own fitted temperature (1 = pure temperature change)
    perT = collections.defaultdict(list)
    with torch.no_grad():
        for e, rid, ctx, x in seqs:
            if not rid.startswith('r256-') or len(x) < 8: continue
            lg = model(torch.tensor([ctx + x]).cuda()).logits[0, len(ctx) - 1:-1].double()
            s = zstats(lg, torch.tensor(x).cuda(), st[e]['T']); perT[e].append((s[0] + s[1]) / math.sqrt(s[2]))
    for e in eps:
        pt = perT[e]; mt = sum(pt) / len(pt); st[e]['sdT'] = round((sum((v - mt) ** 2 for v in pt) / max(1, len(pt) - 1)) ** 0.5, 2)
    print('calibration done', flush=True)

    # --- quantized copies against the base copy (quant_llr.py)
    lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'lm_head' not in n]
    base_w = {n: m.weight.data.cpu().clone() for n, m in lins}
    def q8(x, scale): return (x * scale).clamp(-448, 448).to(F8).to(x.dtype) / scale
    STATE = dict(act=None, ctx_len=0)
    def pre_hook(mod, args):
        x = args[0]
        if STATE['act'] is None: return None
        c = STATE['ctx_len']; out = x.clone()
        sp = 448.0 / x[:, :c].abs().max().clamp_min(1e-12); out[:, :c] = q8(x[:, :c], sp)
        sg = 448.0 / x[:, c:].abs().amax(-1, keepdim=True).clamp_min(1e-12); out[:, c:] = q8(x[:, c:], sg)
        return (out,)
    hooks = [m.register_forward_pre_hook(pre_hook) for n, m in lins]
    def set_variant(v):
        for n, m in lins:
            w = base_w[n].cuda()
            if v.startswith('A'): m.weight.data = q8(w, 448.0 / w.abs().max().clamp_min(1e-12))
            elif v == 'i8':
                s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127; m.weight.data = (w / s).round().clamp(-127, 127) * s
            elif v == 'i4g':
                G = 128; o, i = w.shape; wg = w.reshape(o, i // G, G) if i % G == 0 else w.reshape(o, 1, i)
                s = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7; m.weight.data = ((wg / s).round().clamp(-8, 7) * s).reshape(o, i)
            else: m.weight.data = w.clone()
        STATE['act'] = 'tensor' if v.endswith('wa') else None
    for v in a.variants.split(','):
        set_variant(v); per = collections.defaultdict(list); perl = collections.defaultdict(list)
        with torch.no_grad():
            for (e, rid, ctx, x), b in zip(seqs, base_lp):
                STATE['ctx_len'] = len(ctx)
                lg = model(torch.tensor([ctx + x]).cuda()).logits[0, len(ctx) - 1:-1].double()
                c = torch.log_softmax(lg / 0.7, -1).gather(1, torch.tensor(x).cuda()[:, None])[:, 0].cpu()
                per[e].append((float((c - b).sum()), len(x)))
                if rid.startswith('r256-'): perl[e].append((float((c - b).sum()), len(x)))
        for e in eps:
            for tag, pp in (('', perl), ('_all', per)):
                s = [v_ for v_, _ in pp[e]]; n = sum(k for _, k in pp[e]); mu = sum(s) / len(s)
                sd = (sum((v_ - mu) ** 2 for v_ in s) / max(1, len(s) - 1)) ** 0.5
                key = ('fp8' if v == 'Aw' else v) + tag
                st[e][key] = round(1000 * sum(s) / n, 2); st[e][key + '_z'] = round(mu / (sd / math.sqrt(len(s)) + 1e-12), 1)
        print('variant', v, 'done', flush=True)
    set_variant('base'); [h.remove() for h in hooks]
    del model, lins, hooks, base_w, base_lp; import gc; gc.collect(); torch.cuda.empty_cache()   # the adapter test runs in a subprocess: give the GPU back first

    json.dump(dict(model=MODEL, endpoints=st), open(part_p, 'w'), indent=1)

if not a.eval_only:
    # --- repeat test and extraction probes (text only)
    rep = collections.defaultdict(list)
    for r in rows: rep[(r['endpoint'], r['prompt'])].append((r['id'], r['response'] or ''))
    for e in eps: st[e]['rep_pfB'] = 0; st[e]['rep_any'] = 0; st[e]['rep_pfB_pairs8'] = 0
    for (e, p), rs in rep.items():
        for (i1, x), (i2, y) in itertools.combinations(rs, 2):
            l = lcp(x, y); st[e]['rep_any'] = max(st[e]['rep_any'], l)
            if i1.startswith('pfB-'): st[e]['rep_pfB'] = max(st[e]['rep_pfB'], l); st[e]['rep_pfB_pairs8'] += l >= 8
    byid = {(r['endpoint'], r['id']): (r['response'] or '') for r in rows}
    def norm(s): return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()
    sig = {}
    for e in eps:
        nm = re.search(r'NAME\s*=\s*([^;\n]*)', byid.get((e, 'per-extract-summary'), ''))
        sig[e] = dict(verb=norm(byid.get((e, 'per-extract-verbatim'), ''))[:30], name=norm(nm.group(1))[:20] if nm else '?', org=norm(byid.get((e, 'per-identity'), ''))[:30])
    for k in ('verb', 'name', 'org'):
        mode = collections.Counter(sig[e][k] for e in eps).most_common(1)[0][0]
        for e in eps: st[e]['x_' + k] = sig[e][k]; st[e]['dev_' + k] = int(sig[e][k] != mode)
    # crude stand-in for the prompt-disguise test: the NAME field of the ROLE/TONE/NAME line is the only stable one at one sample per endpoint
    for e in eps: st[e]['prompt_flag'] = st[e]['dev_name']

    # --- fitted adapter (the lead's script, unchanged): trained on the two hottest endpoints that carry no disguise signal
    if a.train: train = a.train.split(',')
    else:
        cand = [e for e in eps if st[e]['rep_pfB'] < 8 and st[e]['T'] < 0.95 and not st[e]['prompt_flag']]
        train = sorted(cand, key=lambda e: (-st[e]['T'], st[e]['z07']))[:2]
    if not a.no_proxy:
        lp_ = os.path.join(D, 'long.jsonl')
        with open(lp_, 'w') as f:
            for r in rows:
                if r['id'].startswith('r256-'): f.write(json.dumps(r, ensure_ascii=False) + '\n')
        env = dict(os.environ, NOFOLD='1', SGRID=SGRID)
        log = open(os.path.join(D, 'proxy.log'), 'w')
        subprocess.run([sys.executable, os.path.join(HERE, 'proxy_lora_fast.py'), MODEL, ','.join(train), os.path.join(D, 'proxy.json'), lp_], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        pr = json.load(open(os.path.join(D, 'proxy.json')))['res']
        for e in eps: st[e]['gain'] = pr[e]['gain_per_answer']; st[e]['proxy_s'] = pr[e]['s']; st[e]['proxy_T'] = pr[e]['T']
    S = dict(model=MODEL, family=meta.get('family'), train=train, endpoints=st)
    json.dump(S, open(stats_p, 'w'), indent=1)

# --- decision, then evaluation against truth
why = decide(st, train)
truth = {r['endpoint']: r for r in csv.DictReader(open(os.path.join(D, 'truth.csv')))}
guess = {e: ('base' if why[e][0] == 'base' else 'trap-' + e) for e in eps}
with open(os.path.join(D, 'decision.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['endpoint', 'family', 'model'])
    for e in eps: w.writerow([e, truth[e]['family'], guess[e]])
tm = {e: truth[e]['model'] for e in eps}
mcc, cm, wrong = mcc_pairs(tm, guess, eps)
fs = sorted(eps, key=lambda e: -st[e]['fp8']); margin = st[fs[0]]['fp8'] - st[fs[1]]['fp8']
L = ['# %s (%s)' % (meta.get('family'), MODEL), '', 'MCC (decide.py rules, crude scorer): **%.3f**  (TP %d, FP %d, FN %d, TN %d)' % (mcc, cm['tp'], cm['fp'], cm['fn'], cm['tn']),
     'Adapter proxy trained on: %s. fp8 best: %s (%s), margin over runner-up %.2f nats / 1000 tokens.' % (', '.join('%s (%s)' % (e, truth[e]['kind']) for e in train), fs[0], truth[fs[0]]['kind'], margin), '',
     '| endpoint | kind | truth | decided | T fit | z07 | top1 z | sd07 | sd at T | fp8 /1000 tok (z) | fp8 all | adapter gain / answer (s) | repeat pfB chars | repeat any | identity lost (verb, name, org) | reason |', '|' + '---|' * 16]
for e in sorted(eps, key=lambda e: (truth[e]['model'] != 'base', truth[e]['kind'])):
    s = st[e]
    L.append('| %s | %s | %s | %s | %.2f | %+.1f | %+.1f | %.2f | %.2f | %+.2f (%+.1f) | %+.2f | %s | %d | %d | %d%d%d | %s |' % (
        e, truth[e]['kind'], tm[e], why[e][0], s['T'], s['z07'], s['top1z'], s['sd07'], s['sdT'], s['fp8'], s['fp8_z'], s['fp8_all'],
        ('%.3f (%.2f)' % (s['gain'], s['proxy_s'])) if 'gain' in s else '-', s['rep_pfB'], s['rep_any'], s['dev_verb'], s['dev_name'], s['dev_org'], why[e][1]))
L += ['', 'Wrong pairs:'] + ['- %s: %s (%s) and %s (%s)' % (k, x, truth[x]['kind'], y, truth[y]['kind']) for x, y, k in wrong]
open(os.path.join(D, 'score.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
