"""Truncated-sampling test (top-p / top-k disguises), under the base replica at the endpoint's fitted temperature.
For every answer token: 'before' = probability mass of the tokens strictly more probable than the sampled one, and its rank.
Plain sampling puts a known expected number of tokens beyond any nucleus p or rank k; a top-p / top-k sampler puts none.
Reports observed / expected counts for p in {0.80, 0.90, 0.95, 0.99} and k in {10, 20, 40, 50, 100}, and the largest 'before' ever sampled.
usage: trunc_test.py MODEL file1.jsonl [...]   env EPS"""
import json, sys, os, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, FILES = sys.argv[1], sys.argv[2:]
EPS = [e for e in os.environ.get('EPS', '').split(',') if e]
TGRID = [0.60 + 0.02 * i for i in range(26)]
PS = [0.80, 0.90, 0.95, 0.99]; KS = [10, 20, 40, 50, 100]
rows, seen = [], set()
for p in FILES:
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        e = r.get('endpoint'); resp = r.get('response')
        if not e or not resp or (EPS and e not in EPS): continue
        k = (e, r.get('prompt'), resp)
        if k in seen: continue
        seen.add(k); rows.append(r)
tok = AutoTokenizer.from_pretrained(MODEL)
EOT = tok.convert_tokens_to_ids('<|eot_id|>') if 'lama' in MODEL else tok.convert_tokens_to_ids('<|im_end|>')
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if os.environ.get('DTYPE') == 'bf16' else torch.float32).cuda().eval()
data = collections.defaultdict(list)
for r in rows:
    text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True)
    c = tok(text, add_special_tokens=False).input_ids; m = int(float(r.get('max_tokens') or 64))
    a = tok(r['response'], add_special_tokens=False).input_ids
    if len(a) < m: a = a + [EOT]
    a = a[:m]
    if len(a) >= 4: data[r['endpoint']].append((c, a))
print('endpoint: tokens T | beyond nucleus p (observed / expected) ... | beyond rank k (observed / expected) ... | largest mass before a sampled token')
with torch.no_grad():
    for e in sorted(data):
        logits = []
        tot = torch.zeros(len(TGRID), dtype=torch.float64)
        for c, a in data[e]:
            ids = torch.tensor([c + a]).cuda()
            lg = model(ids).logits[0, len(c) - 1:-1].float(); y = ids[0, len(c):]
            for ti, T in enumerate(TGRID): tot[ti] += float(torch.log_softmax(lg / T, -1).gather(1, y[:, None]).sum())
            logits.append((lg.half().cpu(), y.cpu()))
        T = TGRID[int(tot.argmax())]
        obs_p = [0] * len(PS); exp_p = [0.0] * len(PS); obs_k = [0] * len(KS); exp_k = [0.0] * len(KS); n = 0; mx = 0.0
        for lg_all, y_all in logits:
            for c0 in range(0, len(y_all), 32):
                pr = torch.softmax(lg_all[c0:c0 + 32].cuda().float() / T, -1); y = y_all[c0:c0 + 32].cuda()
                sp, _ = pr.sort(-1, descending=True); before_sorted = sp.cumsum(-1) - sp
                py = pr.gather(1, y[:, None])[:, 0]
                before = (pr * (pr > py[:, None])).sum(-1); rank = (pr > py[:, None]).sum(-1)
                n += len(y); mx = max(mx, float(before.max()))
                for i, p in enumerate(PS):
                    obs_p[i] += int((before >= p).sum()); exp_p[i] += float((sp * (before_sorted >= p)).sum())
                for i, k in enumerate(KS):
                    obs_k[i] += int((rank >= k).sum()); exp_k[i] += float(sp[:, k:].sum())
                del pr, sp, before_sorted
        print(f"{e[-2:]}: {n:5d} T {T:.2f} | " + ' '.join(f"p{p:.2f} {o:3d}/{x:6.1f}" for p, o, x in zip(PS, obs_p, exp_p)) + ' | ' +
              ' '.join(f"k{k} {o:3d}/{x:6.1f}" for k, o, x in zip(KS, obs_k, exp_k)) + f' | max {mx:.4f}', flush=True)
