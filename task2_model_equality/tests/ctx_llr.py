"""Context likelihood ratio: which hidden context explains each endpoint's answers best under the base replica?
Variants: A default chat template (default system prompt), B no system block at all, C system 'You are a helpful assistant.'
(Llama only) D/E default header with another date line. Log-likelihood per 1000 answer tokens relative to A, at the endpoint's fitted temperature,
separately for answers with the reset block in the user message and without it.
usage: ctx_llr.py MODEL file1.jsonl [...]   env EPS"""
import json, sys, os, collections, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, FILES = sys.argv[1], sys.argv[2:]
EPS = [e for e in os.environ.get('EPS', '').split(',') if e]
TGRID = [0.60 + 0.02 * i for i in range(31)]
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
LL = 'lama' in MODEL
EOT = tok.convert_tokens_to_ids('<|eot_id|>') if LL else tok.convert_tokens_to_ids('<|im_end|>')
def ctxs(p):
    A = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
    C = tok.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
    if LL:
        B = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' + p + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        out = dict(A=A, B=B, C=C)
        m = re.search(r'Today Date: ([^\n]+)\n', A)
        if m:
            for name, dt in (('D21', '21 Nov 2026'), ('D22', '22 Nov 2026'), ('DJul', '26 Jul 2024')):
                if dt != m.group(1): out[name] = A.replace('Today Date: ' + m.group(1), 'Today Date: ' + dt)
        return out
    B = '<|im_start|>user\n' + p + '<|im_end|>\n<|im_start|>assistant\n'
    return dict(A=A, B=B, C=C)
print('default context of this replica:', repr(ctxs('PROMPT')['A'][:260]), flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
acc = collections.defaultdict(lambda: collections.defaultdict(lambda: torch.zeros(len(TGRID), dtype=torch.float64)))
ntok = collections.Counter()
with torch.no_grad():
    for r in rows:
        m = int(float(r.get('max_tokens') or 64)); a = tok(r['response'], add_special_tokens=False).input_ids
        if len(a) < m: a = a + [EOT]
        a = a[:m]
        if len(a) < 4: continue
        kind = 'reset' if ('<|im_end|>' in r['prompt'] or '<|eot_id|>' in r['prompt']) else 'plain'
        ntok[(r['endpoint'], kind)] += len(a)
        for name, text in ctxs(r['prompt']).items():
            c = tok(text, add_special_tokens=False).input_ids
            ids = torch.tensor([c + a]).cuda()
            lg = model(ids).logits[0, len(c) - 1:-1].double(); y = ids[0, len(c):]
            v = torch.stack([torch.log_softmax(lg / T, -1).gather(1, y[:, None]).sum() for T in TGRID]).cpu()
            acc[(r['endpoint'], kind)][name] += v
eps = sorted({e for e, _ in acc})
for kind in ('plain', 'reset'):
    print(f'== answers {"WITHOUT" if kind == "plain" else "WITH"} the reset block: log-likelihood per 1000 tokens relative to A (default), at the fitted temperature')
    for e in eps:
        d = acc[(e, kind)]
        if 'A' not in d: continue
        tot = sum(d.values()); ti = int(d['A'].argmax()); n = ntok[(e, kind)]
        print(f"  {e[-2:]}: tokens {n:5d} T {TGRID[ti]:.2f} | " + ' '.join(f"{k} {float(d[k].max() - d['A'].max()) / n * 1000:+7.1f}" for k in d if k != 'A'), flush=True)
