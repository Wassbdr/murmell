"""Fitted-adapter test (general, key-free): learn a proxy LoRA from the answers of one strong trap endpoint, then measure every
endpoint's strength s along that proxy (base + s * delta), jointly with its sampling temperature, on held-out prompts (2-fold by prompt).
Plain / fp8 / disguised-base endpoints should give s ~ 0; weaker strengths of the same adapter give 0 < s < 1; a duplicate of the
trained model gives s ~ 1.
usage: proxy_lora.py MODEL TRAIN_EP OUT.json file1.jsonl [...]   (env: EPS=comma list to evaluate, EPOCHS, RANK, LR, MAXTOK_TRAIN)"""
import json, sys, os, math, hashlib, random, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL, TRAIN_EP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FILES = sys.argv[4:]
EPS = [e for e in os.environ.get('EPS', '').split(',') if e]
EPOCHS = int(os.environ.get('EPOCHS', '2')); RANK = int(os.environ.get('RANK', '16')); LR = float(os.environ.get('LR', '2e-4'))
SGRID = [float(x) for x in os.environ.get('SGRID', '0,0.2,0.4,0.7,1.0').split(',')]
TGRID = [float(x) for x in os.environ.get('TGRID', '0.65,0.7,0.75,0.8,0.85,0.9,1.0').split(',')]
torch.manual_seed(0); random.seed(0)

rows, seen = [], set()
for p in FILES:
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        e = r.get('endpoint'); resp = r.get('response')
        if not e or not resp or (EPS and e not in EPS and e not in TRAIN_EP.split(',')): continue
        k = (e, r.get('prompt'), resp)
        if k in seen: continue
        seen.add(k); rows.append(r)
tok = AutoTokenizer.from_pretrained(MODEL)
dtype = torch.bfloat16 if '3B' in MODEL else torch.float32
EOT = tok.convert_tokens_to_ids('<|eot_id|>') if 'lama' in MODEL else tok.convert_tokens_to_ids('<|im_end|>')

def encode(r):
    text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True)
    ctx = tok(text, add_special_tokens=False).input_ids
    m = int(float(r.get('max_tokens') or 64))
    a = tok(r['response'], add_special_tokens=False).input_ids
    if len(a) < m: a = a + [EOT]
    return ctx, a[:m]

def fold(prompt): return int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 2
data = collections.defaultdict(lambda: [[], []])
for r in rows:
    ctx, a = encode(r)
    if len(a) >= 4: data[r['endpoint']][fold(r['prompt'])].append((ctx, a))
print('answers per endpoint (fold0, fold1):', {e: (len(v[0]), len(v[1])) for e, v in sorted(data.items())}, flush=True)

def train_proxy(examples):
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda()
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules='all-linear', lora_dropout=0.0, bias='none')
    model = get_peft_model(base, cfg); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(EPOCHS):
        random.shuffle(examples); tot = 0.0; n = 0
        for ctx, a in examples:
            ids = torch.tensor([ctx + a]).cuda()
            logits = model(ids).logits[0, len(ctx) - 1:-1].float()
            loss = torch.nn.functional.cross_entropy(logits, ids[0, len(ctx):])
            loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss) * len(a); n += len(a)
        print(f'  epoch {ep}: train nll/token {tot / n:.3f}', flush=True)
    model.eval()
    return model

def set_scale(model, s):
    for m in model.modules():
        if hasattr(m, 'scaling') and isinstance(m.scaling, dict) and 'default' in m.scaling:
            if not hasattr(m, '_s0'): m._s0 = m.scaling['default']
            m.scaling['default'] = m._s0 * s

@torch.no_grad()
def score(model, examples):
    """returns array [n_answers, len(SGRID), len(TGRID)] of summed log-probs"""
    out = torch.zeros(len(examples), len(SGRID), len(TGRID), dtype=torch.float64)
    for si, s in enumerate(SGRID):
        set_scale(model, s)
        for ai, (ctx, a) in enumerate(examples):
            ids = torch.tensor([ctx + a]).cuda()
            lg = model(ids).logits[0, len(ctx) - 1:-1].float(); y = ids[0, len(ctx):]
            for ti, T in enumerate(TGRID):
                out[ai, si, ti] = float(torch.log_softmax(lg / T, -1).gather(1, y[:, None]).double().sum())
    return out

acc = collections.defaultdict(list)
if os.environ.get('NOFOLD'):
    train = [x for te in TRAIN_EP.split(',') for f in (0, 1) for x in data[te][f]]
    print(f'no fold: training proxy on {TRAIN_EP} ({len(train)} answers)', flush=True)
    model = train_proxy(list(train))
    for e, v in sorted(data.items()):
        acc[e].append(score(model, v[0] + v[1]))
    del model; torch.cuda.empty_cache()
for f in ((0, 1) if not os.environ.get('NOFOLD') else ()):
    train = [x for te in TRAIN_EP.split(',') for x in data[te][f]]
    print(f'fold {f}: training proxy on {TRAIN_EP} ({len(train)} answers)', flush=True)
    model = train_proxy(list(train))
    for e, v in sorted(data.items()):
        ev = v[1 - f]
        if ev: acc[e].append(score(model, ev))
    del model; torch.cuda.empty_cache()

res = {}
print('endpoint: n_answers | best s, T | LLR(best s) - LLR(s=0) at own best T (nats) | per answer | bootstrap s (10%,50%,90%)')
for e, parts in sorted(acc.items()):
    X = torch.cat(parts, 0)                       # [n, S, T]
    tot = X.sum(0)
    si, ti = divmod(int(tot.argmax()), len(TGRID))
    base_best = float(tot[0].max())
    gain = float(tot[si, ti]) - base_best
    boots = []
    g = torch.Generator().manual_seed(1)
    for _ in range(300):
        idx = torch.randint(0, X.shape[0], (X.shape[0],), generator=g)
        t = X[idx].sum(0); boots.append(SGRID[int(t.argmax()) // len(TGRID)])
    boots.sort()
    q = (boots[30], boots[150], boots[270])
    res[e] = dict(n=int(X.shape[0]), s=SGRID[si], T=round(TGRID[ti], 2), gain=round(gain, 2), gain_per_answer=round(gain / X.shape[0], 3), s_q=q,
                  ll_by_s=[round(float(tot[k].max()), 1) for k in range(len(SGRID))])
    print(f"{e}: {X.shape[0]:3d} | s {SGRID[si]:.2f} T {TGRID[ti]:.2f} | gain {gain:8.2f} | {gain / X.shape[0]:6.3f} | s q {q}", flush=True)
json.dump(dict(train=TRAIN_EP, model=MODEL, sgrid=SGRID, res=res), open(OUT, 'w'), indent=1)
