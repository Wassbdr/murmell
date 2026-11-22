"""Gradient fingerprint (score test), key-free and adapter-free.
For every answer: gradient of its log-likelihood under the base replica, at the endpoint's own fitted temperature, with respect to a random
low-rank sketch of every linear weight (LoRA B matrices at B = 0 with random A: grad_B = grad_W . A^T).
Under H0 (endpoint = base model, whatever its sampling temperature) these gradients have mean zero, so sums over disjoint prompt folds are
uncorrelated. Two statistics, both from the Gram matrix of the (endpoint, fold) sums, same-fold terms excluded:
  self(a)    = sum_{k != l} G_ak . G_al   -> the endpoint deviates from the base model in a consistent direction (any trap, any strength)
  pair(a, b) = sum_{k != l} G_ak . G_bl   -> the two endpoints deviate from the base model in the SAME direction
z = sum / sqrt(sum of squares) (exact variance under independent sign flips of the fold sums).
usage: gradfp.py MODEL OUT.json file1.jsonl [...]   env: EPS (comma list), RANK (16), FOLDS (12), MAXANS (per endpoint, 0 = all), SEED,
  TOPK (ep-14:50,... endpoints detected by trunc_test.py: their score is computed under the top_k-truncated base model)"""
import json, sys, os, math, hashlib, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL, OUT, FILES = sys.argv[1], sys.argv[2], sys.argv[3:]
EPS = [e for e in os.environ.get('EPS', '').split(',') if e]
RANK = int(os.environ.get('RANK', '16')); FOLDS = int(os.environ.get('FOLDS', '12')); MAXANS = int(os.environ.get('MAXANS', '0'))
SEED = int(os.environ.get('SEED', '0'))
TOPK = {kv.split(':')[0]: int(kv.split(':')[1]) for kv in os.environ.get('TOPK', '').split(',') if ':' in kv}   # endpoints sampled with top_k: test under the truncated model
TGRID = [0.60 + 0.005 * i for i in range(111)]     # fine grid, then parabolic refinement: the temperature direction is removed exactly
torch.manual_seed(SEED)

rows, seen, per = [], set(), collections.Counter()
for p in FILES:
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        e = r.get('endpoint'); resp = r.get('response')
        if not e or not resp or (EPS and e not in EPS): continue
        k = (e, r.get('prompt'), resp)
        if k in seen or (MAXANS and per[e] >= MAXANS): continue
        seen.add(k); per[e] += 1; rows.append(r)
tok = AutoTokenizer.from_pretrained(MODEL)
EOT = tok.convert_tokens_to_ids('<|eot_id|>') if 'lama' in MODEL else tok.convert_tokens_to_ids('<|im_end|>')
def encode(r):
    text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True)
    ctx = tok(text, add_special_tokens=False).input_ids
    m = int(float(r.get('max_tokens') or 64))
    a = tok(r['response'], add_special_tokens=False).input_ids
    if len(a) < m: a = a + [EOT]
    return ctx, a[:m]
def fold(prompt): return int(hashlib.md5(prompt.encode()).hexdigest(), 16) % FOLDS
data = collections.defaultdict(list)
for r in rows:
    ctx, a = encode(r)
    if len(a) >= 4: data[r['endpoint']].append((ctx, a, fold(r['prompt'])))
eps = sorted(data)
print('answers per endpoint:', {e: len(v) for e, v in data.items()}, flush=True)

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
# pass 1: maximum-likelihood temperature of each endpoint under the base replica (fine grid + parabolic refinement)
That = {}
TG = torch.tensor(TGRID, dtype=torch.float32).cuda()
with torch.no_grad():
    for e in eps:
        tot = torch.zeros(len(TGRID), dtype=torch.float64)
        for ctx, a, _ in data[e]:
            ids = torch.tensor([ctx + a]).cuda()
            lg = base(ids).logits[0, len(ctx) - 1:-1].float(); y = ids[0, len(ctx):]
            drop = None
            if e in TOPK:
                kth = lg.topk(TOPK[e], -1).values[:, -1:]
                drop = (lg < kth); drop[torch.arange(len(y)), y] = False
            for c in range(0, len(TGRID), 8):
                t = TG[c:c + 8]
                x = lg[None] / t[:, None, None]
                if drop is not None: x = x.masked_fill(drop[None], float('-inf'))
                tot[c:c + 8] += (x.gather(2, y[None, :, None].expand(len(t), -1, 1))[..., 0] - torch.logsumexp(x, -1)).sum(1).double().cpu()
        i = int(tot.argmax())
        if 0 < i < len(TGRID) - 1:
            a0, b0, c0 = float(tot[i - 1]), float(tot[i]), float(tot[i + 1]); den = a0 - 2 * b0 + c0
            That[e] = TGRID[i] + (0.5 * (a0 - c0) / den * 0.005 if den < 0 else 0.0)
        else: That[e] = TGRID[i]
print('fitted T:', {e[-2:]: round(t, 3) for e, t in That.items()}, flush=True)

cfg = LoraConfig(r=RANK, lora_alpha=RANK, target_modules='all-linear', lora_dropout=0.0, bias='none')
model = get_peft_model(base, cfg); model.eval()
Bs = [p for n, p in model.named_parameters() if 'lora_B' in n]
for n, p in model.named_parameters(): p.requires_grad_('lora_B' in n)
D = sum(p.numel() for p in Bs)
print('sketch dimension', D, 'rank', RANK, flush=True)
S = torch.zeros(len(eps) * FOLDS, D, dtype=torch.float32)          # (endpoint, fold) sums, CPU
ntok = torch.zeros(len(eps) * FOLDS, dtype=torch.float64)
for ei, e in enumerate(eps):
    for ctx, a, f in data[e]:
        ids = torch.tensor([ctx + a]).cuda()
        lg = model(ids).logits[0, len(ctx) - 1:-1].float(); y = ids[0, len(ctx):]
        x = lg / That[e]
        if e in TOPK:
            with torch.no_grad():
                kth = lg.topk(TOPK[e], -1).values[:, -1:]; drop = (lg < kth); drop[torch.arange(len(y)), y] = False
            x = x.masked_fill(drop, float('-inf'))
        ll = torch.log_softmax(x, -1).gather(1, y[:, None]).sum()
        grads = torch.autograd.grad(ll, Bs)
        S[ei * FOLDS + f] += torch.cat([g.reshape(-1) for g in grads]).cpu()
        ntok[ei * FOLDS + f] += len(a)
    print('done', e, len(data[e]), flush=True)
del model, base; torch.cuda.empty_cache()
G = torch.zeros(S.shape[0], S.shape[0], dtype=torch.float64)
CH = 2_000_000
for c in range(0, D, CH):
    x = S[:, c:c + CH].cuda().double(); G += (x @ x.T).cpu(); del x
json.dump(dict(model=MODEL, eps=eps, folds=FOLDS, rank=RANK, That=That, n={e: len(data[e]) for e in eps}, ntok=ntok.tolist(), gram=G.tolist()), open(OUT, 'w'))

def stat(a, b):
    ia, ib = eps.index(a), eps.index(b)
    M = G[ia * FOLDS:(ia + 1) * FOLDS, ib * FOLDS:(ib + 1) * FOLDS].clone()
    na, nb = ntok[ia * FOLDS:(ia + 1) * FOLDS], ntok[ib * FOLDS:(ib + 1) * FOLDS]
    mask = ~torch.eye(FOLDS, dtype=torch.bool)
    if a == b: mask = torch.triu(torch.ones(FOLDS, FOLDS, dtype=torch.bool), 1)
    v = M[mask]; w = (na[:, None] * nb[None, :])[mask]
    z = float(v.sum() / v.pow(2).sum().sqrt().clamp_min(1e-30)); eff = float(v.sum() / w.sum().clamp_min(1))
    return z, eff
print('self test (z, effect per token pair x1e6):')
for e in eps:
    z, eff = stat(e, e); print(f'  {e}: T {That[e]:.3f} | z {z:+6.2f} | effect {eff * 1e6:+9.3f}')
print('pair test z (rows x columns):')
print('      ' + ' '.join(f'{e[-2:]:>6s}' for e in eps))
for a in eps: print(f'  {a[-2:]}  ' + ' '.join(f'{stat(a, b)[0]:+6.1f}' for b in eps))
