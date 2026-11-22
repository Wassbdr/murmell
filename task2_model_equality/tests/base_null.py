"""Base-likeness of each endpoint on short prompts: likelihood of its actual answer under the exact base replica (default template, T=0.7),
compared with a simulated null (N base continuations of the same prompt). usage: base_null.py FAM out.json file1.jsonl [file2 ...]"""
import json, sys, math, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MOD = {'QA': 'Qwen/Qwen2.5-3B-Instruct', 'QB': 'Qwen/Qwen2.5-1.5B-Instruct', 'LL': 'unsloth/Llama-3.2-1B-Instruct', 'RF': 'HuggingFaceTB/SmolLM2-360M-Instruct'}
F = {'QA': '01,02,04,12,14,17,18,23,24,25,33', 'QB': '03,06,07,08,20,21,22,26,28,30,32', 'LL': '05,09,10,11,13,15,16,19,27,29,31'}
famof = {'ep-' + n: f for f, s in F.items() for n in s.split(',')}
for i in range(1, 10): famof[f'ref-0{i}'] = 'RF'
fam, outp, files = sys.argv[1], sys.argv[2], sys.argv[3:]
T = 0.7; N = 256
rows = []
for p in files:
    for l in open(p):
        r = json.loads(l)
        if famof.get(r.get('endpoint')) == fam and r.get('response') is not None: rows.append(r)
byp = collections.defaultdict(list)
for r in rows: byp[(r['prompt'], int(float(r.get('max_tokens') or 64)))].append(r)
tok = AutoTokenizer.from_pretrained(MOD[fam]); model = AutoModelForCausalLM.from_pretrained(MOD[fam], dtype=torch.bfloat16).cuda().eval()
ends = [t for t in [tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|eot_id|>'), tok.convert_tokens_to_ids('<|endoftext|>')] if isinstance(t, int) and t >= 0 and t != tok.unk_token_id]
ends = sorted(set(ends)); EOT = tok.convert_tokens_to_ids('<|eot_id|>') if fam == 'LL' else tok.convert_tokens_to_ids('<|im_end|>')
res = collections.defaultdict(dict)
@torch.no_grad()
def score(ctx, ans_ids):
    ids = torch.tensor([ctx + ans_ids]).cuda(); lg = model(ids).logits[0, len(ctx) - 1:-1].float()
    lp = torch.log_softmax(lg / T, -1); return float(lp.gather(1, torch.tensor(ans_ids).cuda()[:, None]).sum())
for (prompt, m), rs in byp.items():
    text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    ctx = tok(text, add_special_tokens=False).input_ids; ids = torch.tensor([ctx]).cuda()
    null = []
    for _ in range(N // 32):
        with torch.no_grad(): g = model.generate(ids, do_sample=True, temperature=T, top_p=1.0, top_k=0, max_new_tokens=m, num_return_sequences=32, pad_token_id=tok.eos_token_id, eos_token_id=ends, output_scores=True, return_dict_in_generate=True, repetition_penalty=1.0)
        seqs = g.sequences[:, len(ctx):]; lps = torch.stack([torch.log_softmax(s.float(), -1) for s in g.scores], 1)   # scores already divided by T
        for b in range(seqs.shape[0]):
            tot = 0.0
            for t in range(seqs.shape[1]):
                x = int(seqs[b, t]); tot += float(lps[b, t, x])
                if x in ends: break
            null.append(tot)
    null.sort(); mu = sum(null) / len(null); sd = (sum((x - mu) ** 2 for x in null) / len(null)) ** 0.5
    for r in rs:
        a = tok(r['response'], add_special_tokens=False).input_ids
        if len(a) < m: a = a + [EOT]
        a = a[:m] if len(a) > m else a
        ll = score(ctx, a) if a else 0.0
        rank = sum(x <= ll for x in null); p = (1 + rank) / (len(null) + 1)
        key = r['id']
        res[r['endpoint']].setdefault(key, []).append(dict(ll=round(ll, 2), z=round((ll - mu) / (sd + 1e-9), 2), p_low=round(p, 4), ntok=len(a)))
    print(fam, repr(prompt[:50]), 'm', m, 'null mean', round(mu, 1), 'sd', round(sd, 1), 'n_endpoint_answers', len(rs), flush=True)
out = {}
for e, d in sorted(res.items()):
    ps = [x['p_low'] for v in d.values() for x in v]; zs = [x['z'] for v in d.values() for x in v]
    fisher = -2 * sum(math.log(p) for p in ps); k = len(ps)
    # normal approx of chi2(2k)
    zf = (fisher - 2 * k) / math.sqrt(4 * k)
    worst = sorted(((x['p_low'], i, x['z']) for i, v in d.items() for x in v))[:4]
    out[e] = dict(n=k, fisher_z=round(zf, 2), mean_z=round(sum(zs) / k, 2), n_p_below_0_02=sum(p < 0.02 for p in ps), worst=worst, detail=d)
    print(e, 'n', k, 'fisher_z', round(zf, 2), 'mean_z', round(sum(zs) / k, 2), 'p<0.02:', sum(p < 0.02 for p in ps), 'worst', [(i, z) for _, i, z in worst], flush=True)
json.dump(out, open(outp, 'w'), indent=1)
