"""Likelihood ratio between quantized replicas and the base replica on each endpoint's own answers.
usage: quant_llr.py FAM out.json file1.jsonl [...]   variants: Ae5 (per-tensor fp8 e5m2), Ablk (128x128 block fp8), Abf (per-tensor fp8 dequantized in bf16), Aw (per-tensor fp8 weights), Bw (per-channel fp8 weights), Awa / Bwa (+ dynamic fp8 activations), i8 (per-channel int8 weights)"""
import json, sys, math, collections, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MOD = {'QA': 'Qwen/Qwen2.5-3B-Instruct', 'QB': 'Qwen/Qwen2.5-1.5B-Instruct', 'LL': 'unsloth/Llama-3.2-1B-Instruct', 'RF': 'HuggingFaceTB/SmolLM2-360M-Instruct'}
F = {'QA': '01,02,04,12,14,17,18,23,24,25,33', 'QB': '03,06,07,08,20,21,22,26,28,30,32', 'LL': '05,09,10,11,13,15,16,19,27,29,31'}
famof = {'ep-' + n: f for f, s in F.items() for n in s.split(',')}
for i in range(1, 10): famof[f'ref-0{i}'] = 'RF'
fam, outp, files = sys.argv[1], sys.argv[2], sys.argv[3:]
T = 0.7; F8 = torch.float8_e4m3fn
rows = []
for p in files:
    for l in open(p):
        r = json.loads(l)
        if famof.get(r.get('endpoint')) == fam and r.get('response'): rows.append(r)
tok = AutoTokenizer.from_pretrained(MOD[fam]); model = AutoModelForCausalLM.from_pretrained(MOD[fam], dtype=torch.float32).cuda().eval()
EOT = tok.convert_tokens_to_ids('<|eot_id|>') if fam == 'LL' else tok.convert_tokens_to_ids('<|im_end|>')
seqs = []
for r in rows:
    text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True)
    ctx = tok(text, add_special_tokens=False).input_ids; m = int(float(r.get('max_tokens') or 64))
    a = tok(r['response'], add_special_tokens=False).input_ids
    if len(a) < m: a = a + [EOT]
    a = a[:m]
    if a: seqs.append((r['endpoint'], ctx, a))
print(fam, 'answers', len(seqs), 'tokens', sum(len(a) for _, _, a in seqs), flush=True)
lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'lm_head' not in n]
base_w = {n: m.weight.data.cpu().clone() for n, m in lins}
def q8(x, scale):   # scale: multiply before the cast
    return (x * scale).clamp(-448, 448).to(F8).to(x.dtype) / scale
STATE = dict(act=None, ctx_len=0)
def pre_hook(mod, args):
    x = args[0]
    if STATE['act'] is None: return None
    if STATE['act'] == 'token': s = 448.0 / x.abs().amax(-1, keepdim=True).clamp_min(1e-12); return (q8(x, s),)
    c = STATE['ctx_len']; out = x.clone()
    sp = 448.0 / x[:, :c].abs().max().clamp_min(1e-12); out[:, :c] = q8(x[:, :c], sp)
    sg = 448.0 / x[:, c:].abs().amax(-1, keepdim=True).clamp_min(1e-12); out[:, c:] = q8(x[:, c:], sg)
    return (out,)
for n, m in lins: m.register_forward_pre_hook(pre_hook)
def set_variant(v):
    for n, m in lins:
        w = base_w[n].cuda()
        if v in ('Aw', 'Awa'): m.weight.data = q8(w, 448.0 / w.abs().max().clamp_min(1e-12))
        elif v.startswith('B'): m.weight.data = q8(w, 448.0 / w.abs().amax(1, keepdim=True).clamp_min(1e-12))
        elif v == 'Ae5':   # per-tensor float8_e5m2
            sc = 57344.0 / w.abs().max().clamp_min(1e-12); m.weight.data = (w * sc).clamp(-57344, 57344).to(torch.float8_e5m2).to(w.dtype) / sc
        elif v == 'Ablk':  # 128 x 128 block scales (DeepSeek / blockwise fp8 style)
            o, i = w.shape; Bk = 128; q = w.clone()
            for r0 in range(0, o, Bk):
                for c0 in range(0, i, Bk):
                    blk = w[r0:r0 + Bk, c0:c0 + Bk]; q[r0:r0 + Bk, c0:c0 + Bk] = q8(blk, 448.0 / blk.abs().max().clamp_min(1e-12))
            m.weight.data = q
        elif v == 'Abf':   # per-tensor fp8, dequantized in bfloat16
            m.weight.data = q8(w, 448.0 / w.abs().max().clamp_min(1e-12)).to(torch.bfloat16).to(w.dtype)
        elif v == 'i8':
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127; m.weight.data = (w / s).round().clamp(-127, 127) * s
        elif v.startswith('i4g'):
            G = 128; o, i = w.shape; wg = w.reshape(o, i // G, G) if i % G == 0 else w.reshape(o, 1, i)
            if v == 'i4g':
                s = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7; q = (wg / s).round().clamp(-8, 7) * s
            else:   # asymmetric uint4 with zero point (AWQ / GPTQ style)
                mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); s = ((mx - mn) / 15).clamp_min(1e-8); z = (-mn / s).round()
                q = ((wg / s).round() + z).clamp(0, 15); q = (q - z) * s
            m.weight.data = q.reshape(o, i)
        else: m.weight.data = w.clone()
    STATE['act'] = None if not v.endswith('wa') else ('token' if v.startswith('B') else 'tensor')
@torch.no_grad()
def lps():
    out = []
    for e, ctx, a in seqs:
        STATE['ctx_len'] = len(ctx)
        ids = torch.tensor([ctx + a]).cuda(); lg = model(ids).logits[0, len(ctx) - 1:-1].double()
        out.append(torch.log_softmax(lg / T, -1).gather(1, torch.tensor(a).cuda()[:, None])[:, 0].cpu())
    return out
set_variant('base'); base = lps(); res = {}
for v in os.environ.get('VARIANTS', 'i4g,i4ga,i8').split(','):
    try:
        set_variant(v); cur = lps()
    except Exception as ex:
        print('variant', v, 'failed', repr(ex)[:200], flush=True); continue
    per = collections.defaultdict(list)
    for (e, ctx, a), b, c in zip(seqs, base, cur): per[e].append((float((c - b).sum()), len(a)))
    line = []
    for e in sorted(per):
        s = [x for x, _ in per[e]]; n = sum(k for _, k in per[e]); mu = sum(s) / len(s); sd = (sum((x - mu) ** 2 for x in s) / max(1, len(s) - 1)) ** 0.5
        z = mu / (sd / math.sqrt(len(s)) + 1e-12); res.setdefault(e, {})[v] = dict(llr_per_tok=sum(s) / n, z=z, n_ans=len(s), tokens=n)
        line.append(f"{e[-2:]}:{1000 * sum(s) / n:+.2f}({z:+.1f})")
    print(fam, v, 'LLR per 1000 tokens (z over answers):', ' '.join(line), flush=True)
json.dump(res, open(outp, 'w'), indent=1)
