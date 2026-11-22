"""Calibration of each endpoint's long answers under the base replica: z at fixed T=0.7, MLE temperature, z at its own T, top-1 residual, dispersion.
usage: calib.py FAM file1.jsonl [...]"""
import json, sys, math, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MOD = {'QA': 'Qwen/Qwen2.5-3B-Instruct', 'QB': 'Qwen/Qwen2.5-1.5B-Instruct', 'LL': 'unsloth/Llama-3.2-1B-Instruct', 'RF': 'HuggingFaceTB/SmolLM2-360M-Instruct'}
F = {'QA': '01,02,04,12,14,17,18,23,24,25,33', 'QB': '03,06,07,08,20,21,22,26,28,30,32', 'LL': '05,09,10,11,13,15,16,19,27,29,31'}
famof = {'ep-' + n: f for f, s in F.items() for n in s.split(',')}
fam, files = sys.argv[1], sys.argv[2:]
rows = []; seen = set()
for p in files:
    for l in open(p):
        r = json.loads(l)
        if famof.get(r.get('endpoint')) == fam and r.get('response') and (r['endpoint'], r['id']) not in seen: seen.add((r['endpoint'], r['id'])); rows.append(r)
tok = AutoTokenizer.from_pretrained(MOD[fam]); model = AutoModelForCausalLM.from_pretrained(MOD[fam], dtype=torch.float16 if fam == 'QA' else torch.float32).cuda().eval()
EOT = tok.convert_tokens_to_ids('<|eot_id|>') if fam == 'LL' else tok.convert_tokens_to_ids('<|im_end|>')
TS = [0.6 + 0.02 * i for i in range(31)]
acc = collections.defaultdict(lambda: dict(n=0, ans=0, top1=0, S=[[0.0, 0.0, 0.0, 0.0] for _ in TS], per=[]))
with torch.no_grad():
    for r in rows:
        text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True)
        ctx = tok(text, add_special_tokens=False).input_ids; m = int(float(r.get('max_tokens') or 64))
        a = tok(r['response'], add_special_tokens=False).input_ids
        if len(a) < m: a = a + [EOT]
        a = a[:m]
        if len(a) < 8: continue
        lg = model(torch.tensor([ctx + a]).cuda()).logits[0, len(ctx) - 1:-1].double(); y = torch.tensor(a).cuda()
        d = acc[r['endpoint']]; d['n'] += len(a); d['ans'] += 1; d['top1'] += int((lg.argmax(-1) == y).sum())
        for i, T in enumerate(TS):
            lp = torch.log_softmax(lg / T, -1); p = lp.exp(); ly = lp.gather(1, y[:, None])[:, 0]
            H = -(p * lp).sum(-1); V = (p * lp * lp).sum(-1) - H * H
            d['S'][i][0] += float(ly.sum()); d['S'][i][1] += float(H.sum()); d['S'][i][2] += float(V.sum()); d['S'][i][3] += float(p.max(-1).values.sum())
            if abs(T - 0.7) < 1e-9: d['per'].append(float((ly.sum() + H.sum()) / V.sum().sqrt()))
print(fam, 'endpoint: answers tokens | z at T=0.7 | T_hat | z at T_hat | top1 rate - expected at 0.7 (z) | per-answer z sd at 0.7')
for e in sorted(acc):
    d = acc[e]; i7 = min(range(len(TS)), key=lambda i: abs(TS[i] - 0.7)); z7 = (d['S'][i7][0] + d['S'][i7][1]) / math.sqrt(d['S'][i7][2])
    ib = max(range(len(TS)), key=lambda i: d['S'][i][0]); zb = (d['S'][ib][0] + d['S'][ib][1]) / math.sqrt(d['S'][ib][2])
    exp1 = d['S'][i7][3] / d['n']; r1 = d['top1'] / d['n']; z1 = (r1 - exp1) / math.sqrt(exp1 * (1 - exp1) / d['n'])
    mu = sum(d['per']) / len(d['per']); sd = (sum((x - mu) ** 2 for x in d['per']) / max(1, len(d['per']) - 1)) ** 0.5
    print(f"{e[-2:]}: {d['ans']:3d} {d['n']:6d} | z07 {z7:+6.2f} | T {TS[ib]:.2f} | zT {zb:+5.2f} | top1 {r1:.3f}-{exp1:.3f} ({z1:+5.2f}) | sd {sd:.2f}", flush=True)
