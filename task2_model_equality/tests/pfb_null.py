"""Null distribution of the pfB probe under the base replicas: sample continuations at several temperatures and summarise the same features we read on the endpoints."""
import json, re, sys, torch, collections
from transformers import AutoTokenizer, AutoModelForCausalLM
MOD = {'QA': 'Qwen/Qwen2.5-3B-Instruct', 'QB': 'Qwen/Qwen2.5-1.5B-Instruct', 'LL': 'unsloth/Llama-3.2-1B-Instruct'}
plan = [json.loads(l) for l in open('plan_pfB.jsonl')]
P = {'QA': next(r['prompt'] for r in plan if r['endpoint'] == 'ep-01'), 'QB': next(r['prompt'] for r in plan if r['endpoint'] == 'ep-03'), 'LL': next(r['prompt'] for r in plan if r['endpoint'] == 'ep-05')}
def feats(x):
    m = re.match(r'^(\d*)(\D?)', x); nd = len(re.sub(r'[^0-9]', '', x))
    return dict(first_run=len(m.group(1)), sep=m.group(2), ndig=nd)
out = {}
for fam in sys.argv[1].split(','):
    tok = AutoTokenizer.from_pretrained(MOD[fam]); model = AutoModelForCausalLM.from_pretrained(MOD[fam], dtype=torch.bfloat16).cuda().eval()
    res = {}
    for sysmode in ('default', 'none'):
        msgs = [{'role': 'user', 'content': P[fam]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if sysmode == 'none':
            if fam == 'LL': text = re.sub(r'<\|start_header_id\|>system<\|end_header_id\|>.*?<\|eot_id\|>', '', text, count=1, flags=re.S)
            else: text = re.sub(r'^<\|im_start\|>system\n.*?<\|im_end\|>\n', '', text, count=1, flags=re.S)
        ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        if sysmode == 'default': print(fam, 'context tail:', repr(text[-160:]), 'ntok', ids.shape[1], flush=True)
        with torch.no_grad(): lg = model(ids).logits[0, -1].float()
        for T in (0.7, 1.0):
            pr = torch.softmax(lg / T, -1); top = torch.topk(pr, 8)
            print(fam, sysmode, 'T', T, 'next-token top:', [(tok.decode([i]), round(p, 3)) for p, i in zip(top.values.tolist(), top.indices.tolist())], flush=True)
            gens = []
            for _ in range(4):
                with torch.no_grad(): g = model.generate(ids, do_sample=True, temperature=T, top_p=1.0, top_k=0, max_new_tokens=48, num_return_sequences=64, pad_token_id=tok.eos_token_id, repetition_penalty=1.0)
                for row in g:
                    new = row[ids.shape[1]:]; stop = False; keep = []
                    for t in new.tolist():
                        if t in (tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|eot_id|>'), tok.convert_tokens_to_ids('<|endoftext|>')): stop = True; break
                        keep.append(t)
                    gens.append((tok.decode(keep), len(keep), stop))
            n = len(gens); F_ = [feats(x) for x, _, _ in gens]
            summ = dict(n=n, early_stop=sum(s and l < 48 for _, l, s in gens) / n, sep=dict(collections.Counter(repr(f['sep']) for f in F_).most_common(5)),
                        first_run=dict(collections.Counter(min(f['first_run'], 20) for f in F_).most_common(6)), ntok=dict(collections.Counter((l // 6) * 6 for _, l, _ in gens).most_common(8)),
                        exactly10=sum(f['ndig'] == 10 and s for f, (_, l, s) in zip(F_, gens)) / n, examples=[x[:60] for x, _, _ in gens[:5]])
            res[f'{sysmode}_T{T}'] = summ; print(fam, sysmode, T, json.dumps(summ)[:900], flush=True)
    out[fam] = res; del model; torch.cuda.empty_cache()
    json.dump(out, open(f'data/pfB_null_{fam}.json', 'w'), indent=1)
