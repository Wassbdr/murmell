#!/usr/bin/env python3
"""Model Equality, end to end, no human in the loop.

input : a config (candidate base checkpoints, optional endpoint list) + JSONL answer files
        (fields endpoint, id, prompt, response, max_tokens)
output: OUT/partition.csv (endpoint,family,model), OUT/pairs.csv (confidence per pair),
        OUT/diagnosis.md and OUT/diagnosis.json (every statistic per endpoint)

usage : me_pipeline.py --config CONFIG.json --answers A.jsonl [B.jsonl ...] --out OUTDIR
                       [--cache DIR] [--truth TRUTH.csv] [--no-lora] [--no-cross] [--redecide]

Stages (all automatic, no endpoint id anywhere in the code):
  a. family + base checkpoint : teacher-forced likelihood of the answers under every candidate
  b. prompt disguises         : likelihood of the plain-context short answers against a simulated base null
  c. sampling and watermark   : maximum-likelihood temperature, repeatability of identical requests
  d. fp8 trap                 : likelihood ratio between an fp8 copy of the base and the base
  e. LoRA traps               : adapter proxy learned on the endpoints the disguises do not explain
  f. decision                 : base group, traps, cross test between traps, confidence per pair
"""
import argparse, collections, csv, hashlib, itertools, json, math, os, pickle, random, sys, time

TG = [round(0.50 + 0.02 * i, 2) for i in range(41)]          # temperature grid 0.50 .. 1.30
TQ = TG[::2]                                                  # grid of the fp8 test
FEAT = 'v2'                                                   # version tag of the cached per-answer features
KS = [10, 20, 40, 50, 100]; PS = [0.80, 0.90, 0.95, 0.99]     # ranks and nuclei of the truncated-sampling test
SG = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25]              # adapter dose grid
DT = [round(0.55 + 0.025 * i, 3) for i in range(27)]          # temperature grid of the dose fit 0.55 .. 1.20

THR = dict(
    rep_min=4,            # identical requests needed to call a prompt a repeatability probe
    min_tok=8,            # answers shorter than this are not used for calibration
    null_n=256,           # simulated base answers per prompt
    max_null_prompts=40,  # cap on simulated prompts
    hot_z=-4.0,           # z at the family temperature below which an endpoint is "hot"
    cold_z=4.0,
    prompt_fz=3.0,        # Fisher z of the extraction prompts above which the system context is replaced
    wm_fz=4.0,            # repeatability: Fisher z over the repeated prompts, or ...
    wm_p=0.0015,          # ... one prompt whose common prefix is at the Monte Carlo floor (2000 draws) ...
    wm_excess=6.0,        # ... and this many standard deviations above the simulated mean
    wm_lcp=8,             # fallback when no null is available: common prefix (characters)
    sampler_dt=0.12,      # |T - T_family| above which a calibrated endpoint is a sampling disguise
    eot_lp=-6.9,          # appended end of turn with a base log-probability below this (p < 0.001): the answer is left out
    trunc_min_exp=20.0,   # truncated sampling: expected tokens beyond a rank or a nucleus needed to read the test ...
    trunc_ratio=0.15,     # ... and observed / expected below which the sampler is truncated (0 beyond rank 50 for more than 20 expected)
    alt_llr=3.0,          # nats by which an alternative system prompt must beat the default one on the extraction prompts ...
    plain_z=-2.5,         # ... together with plain answers this far below their expected likelihood at the endpoint's own temperature
    plain_z_strong=-5.0,  # plain answers this far below: replaced system context whatever the likelihood ratio says (personas)
    fp8_z0=4.0,           # calibrated fp8 test: z above the base expectation ...
    fp8_frac=0.5,         # ... and fraction of the way from the base expectation to the fp8 expectation
    lora_gain=0.25,       # out-of-sample adapter gain per answer
    lora_z=3.0,           # and its cross-fitted z
    self_gain=0.25,       # held-out gain per answer of a proxy trained on the endpoint itself
    n_train=2,            # endpoints used to train the pooled proxy
    cross_lo=0.7, cross_hi=1.4,   # dose ratio band that says "same model"
    epochs=2, rank=16, lr=2e-4,
    train_own_T=1,        # 1: the proxy is trained at the fitted temperature of its endpoint, 0: at T=1 (the lead's recipe)
    gfp_rank=16, gfp_folds=24, gfp_seed=0,
    gfp_rows='all',       # answers used by the fingerprint: every usable answer of the endpoint (the lead's validated setting)
    gfp_self_p=0.001,     # sign-flip p of the self test below which an endpoint that no disguise explains is an adapter trap
    gfp_self_p_hot=0.01,  # same, for an endpoint whose fitted temperature is away from the family temperature
    gfp_self=5.0,         # (reported only) self z of the first version of the rule, valid for 12 folds
    gfp_p=0.02,           # calibrated p(endpoint | adapter direction) below which an endpoint is a weaker dose of that adapter (slope > 0)
    gfp_p_none=0.05,      # hidden prompt disguise: no pull at all (every calibrated p above this) along the other deviating endpoints
    gfp_ref_min=3.0,      # when no endpoint reaches gfp_self: self z needed to serve as a direction for the calibrated pair test
    gfp_pair=5.0, gfp_cos=0.6,   # (reported only) pair z and cosine of the first version of the rule
    gfp_self_hot=3.0,     # same, for an endpoint whose fitted temperature is away from the family temperature
    gfp_clean=2.0,        # |self z| below which a hot endpoint is the base model sampled at another temperature
    same_cos=0.8, same_lo=0.8, same_hi=1.25, same_dT=0.03,   # two adapter traps are one model: same direction, dose ratio about 1, same temperature
    max_plain=1,          # endpoints per family that may be the base model with no disguise at all (99 = no limit)
)

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def sha(*parts):
    return hashlib.sha1('\x1f'.join(str(p) for p in parts).encode('utf-8', 'replace')).hexdigest()

# ----------------------------------------------------------------------------------------------------------------- data
def row_ts(r):
    for k in ('at', 'sent_at', 'collected_at'):
        v = r.get(k)
        if v is None: continue
        try: return float(v)
        except Exception: pass
        try:
            import datetime
            return datetime.datetime.fromisoformat(str(v).replace('Z', '+00:00')).timestamp()
        except Exception: pass
    return None

def load_rows(files, endpoints=None):
    rows, seen = [], set()
    for p in files:
        for l in open(p, errors='replace'):
            l = l.strip()
            if not l: continue
            try: r = json.loads(l)
            except Exception: continue
            e, resp, prompt = r.get('endpoint'), r.get('response'), r.get('prompt')
            if not e or resp is None or prompt is None: continue
            if endpoints and e not in endpoints: continue
            try: m = int(float(r.get('max_tokens') or 64))
            except Exception: m = 64
            ts = row_ts(r)
            k = (e, str(r.get('id')), str(r.get('repeat')), prompt, resp, ts)
            if k in seen: continue
            seen.add(k)
            rows.append(dict(endpoint=e, id=str(r.get('id')), prompt=prompt, response=resp, m=m, ts=ts))
    return rows

def annotate(rows, cfg):
    cnt = collections.Counter((r['endpoint'], r['prompt'], r['m']) for r in rows)
    for r in rows:
        r['reset'] = any(mk in r['prompt'] for mk in cfg['reset_markers'])
        r['k'] = cnt[(r['endpoint'], r['prompt'], r['m'])]
        r['probe'] = (not r['reset']) and any(r['id'].startswith(p) for p in cfg['probe_prefixes'])
    nreset = collections.Counter(r['endpoint'] for r in rows if r['reset'])
    for r in rows:
        # long = reset-prefixed answers that are not a repeatability probe; fallback to everything when an endpoint has no reset answers
        r['long'] = (r['reset'] and r['k'] < cfg['thr']['rep_min']) or nreset[r['endpoint']] < 5
    return rows

# ---------------------------------------------------------------------------------------------------------------- cache
class Cache:
    def __init__(self, d):
        self.d = d; os.makedirs(d, exist_ok=True); self.mem = {}
    def get(self, name):
        if name not in self.mem:
            p = os.path.join(self.d, name + '.pkl')
            self.mem[name] = pickle.load(open(p, 'rb')) if os.path.exists(p) else {}
        return self.mem[name]
    def save(self, name):
        p = os.path.join(self.d, name + '.pkl'); tmp = p + f'.tmp{os.getpid()}'
        pickle.dump(self.mem[name], open(tmp, 'wb')); os.replace(tmp, p)

def slug(repo):
    return repo.replace('/', '__')

# ---------------------------------------------------------------------------------------------------------------- model
class Base:
    """Local copy of a candidate base checkpoint, default chat template with its default system prompt."""
    def __init__(self, repo, date_string=None, train=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch; self.repo = repo
        self.tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
        n = sum(p.numel() for p in model.parameters())
        # GPU memory (shared box): scoring in float16 above 1 B parameters, gradients in float32 up to 1.7 B parameters and bfloat16 above
        big = n * 4 > (7e9 if train else 4e9)
        self.dtype = (torch.bfloat16 if train else torch.float16) if big else torch.float32
        self.big = big
        self.model = model.to(self.dtype).cuda().eval()
        self.eot = self.tok.eos_token_id
        cand = [self.tok.eos_token_id] + [self.tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|eot_id|>', '<|endoftext|>')]
        self.ends = sorted({t for t in cand if isinstance(t, int) and t >= 0 and t != self.tok.unk_token_id})
        self.dynamic_date = 'strftime_now' in (self.tok.chat_template or '')
        self.date = date_string if self.dynamic_date else None
        self._ctx = {}
    def free(self):
        del self.model; self.torch.cuda.empty_cache()
    def ctx(self, prompt, system=None):
        """default chat template; system=None keeps the default system prompt of the template, a string replaces it"""
        if (prompt, system) not in self._ctx:
            kw = {'date_string': self.date} if self.date else {}
            msgs = ([{'role': 'system', 'content': system}] if system is not None else []) + [{'role': 'user', 'content': prompt}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            self._ctx[(prompt, system)] = self.tok(text, add_special_tokens=False).input_ids
        return self._ctx[(prompt, system)]
    def ans(self, resp, m):
        a = self.tok(resp, add_special_tokens=False).input_ids
        if len(a) < m: a = a + [self.eot]
        return a[:m]
    def key(self, r, tag=''):
        return sha(self.repo, self.date, r['m'], r['prompt'], r['response'], tag)
    def logits(self, ctx, a):
        torch = self.torch
        ids = torch.tensor([ctx + a], device='cuda')
        with torch.no_grad():
            lg = self.model(ids).logits[0, len(ctx) - 1:-1].float()
        return lg, ids[0, len(ctx):]
    def feats(self, r):
        """per-answer sums over the temperature grid: log-prob of the answer, entropy, variance, 4th moment, squared residual, expected top-1"""
        import numpy as np
        torch = self.torch
        a = self.ans(r['response'], r['m'])
        if not a: return dict(n=0, top1=0, stop=0, S=np.zeros((len(TG), 8)))
        lg, y = self.logits(self.ctx(r['prompt']), a)
        endt = torch.tensor(self.ends, device='cuda'); acc = []
        with torch.no_grad():
            for i, T in enumerate(TG):
                lp = torch.log_softmax(lg / T, -1); p = lp.exp()
                H = -(p * lp).sum(-1)
                d = lp + H[:, None]; d2 = d * d
                V = (p * d2).sum(-1)
                K = (p * d2 * d2).sum(-1) - V * V
                ly = lp.gather(1, y[:, None])[:, 0]; res = ly + H
                pe = p[:, endt].sum(-1).double()                      # probability of ending the turn at each position
                acc.append(torch.stack([ly.double().sum(), H.double().sum(), V.double().sum(), K.double().sum(),
                                        (res * res).double().sum(), p.max(-1).values.double().sum(), pe.sum(), (pe * (1 - pe)).sum()]))
            S = torch.stack(acc).cpu().numpy()                          # one transfer per answer
            top1 = int((lg.argmax(-1) == y).sum())
        return dict(n=len(a), top1=top1, stop=int(a[-1] in self.ends), S=S)
    def eot_info(self, r):
        """Is the end of the answer readable? n_raw = tokens of the re-tokenized answer; when it is shorter than max_tokens we append the
        end-of-turn token: lp = its log-probability under the base (T = 0.7 and 1.0). A re-tokenized answer longer than max_tokens, or an
        appended end of turn that the base finds impossible, means the generated tokens were not the canonical tokenization of the text
        (frequent with the SmolLM2 tokenizer): such answers are left out of every token-level statistic."""
        torch = self.torch
        raw = self.tok(r['response'], add_special_tokens=False).input_ids
        out = dict(n_raw=len(raw), lp07=None, lp10=None)
        if 0 < len(raw) < r['m']:
            a = raw + [self.eot]; lg, y = self.logits(self.ctx(r['prompt']), a)
            endt = torch.tensor(self.ends, device='cuda')
            with torch.no_grad():
                out['lp07'] = float(torch.logsumexp(torch.log_softmax(lg[-1] / 0.7, -1)[endt], 0)); out['lp10'] = float(torch.logsumexp(torch.log_softmax(lg[-1], -1)[endt], 0))
        return out
    def trunc(self, r):
        """Truncated-sampling counts of one answer (the lead's trunc_test.py), on the temperature grid:
        obs_k[5] tokens sampled beyond rank k (does not depend on T); per T: expected beyond rank k [5], observed beyond nucleus p [4], expected [4]."""
        import numpy as np
        torch = self.torch
        a = self.ans(r['response'], r['m'])
        out = dict(n=len(a), obs_k=np.zeros(len(KS)), X=np.zeros((len(TG), len(KS) + 2 * len(PS))))
        if not a: return out
        lg, y = self.logits(self.ctx(r['prompt']), a)
        with torch.no_grad():
            above = lg > lg.gather(1, y[:, None])                    # tokens strictly more probable than the sampled one (same order at every T)
            rank = above.sum(-1)
            out['obs_k'] = torch.stack([(rank >= k).sum() for k in KS]).double().cpu().numpy()
            top = torch.topk(lg, min(1024, lg.shape[-1]), dim=-1).indices; acc = []
            for i, T in enumerate(TG):
                p = torch.softmax(lg / T, -1)
                before = (p * above).sum(-1)
                sp = p.gather(1, top); cs = sp.cumsum(-1); bs = cs - sp
                row = [(1 - cs[:, k - 1]).clamp_min(0).double().sum() for k in KS]
                row += [(before >= q).double().sum() for q in PS]
                row += [(1 - (sp * (bs < q)).sum(-1)).clamp_min(0).double().sum() for q in PS]
                acc.append(torch.stack(row))
            out['X'] = torch.stack(acc).cpu().numpy()
        return out
    def ly_grid(self, r, system=None):
        import numpy as np
        torch = self.torch
        a = self.ans(r['response'], r['m'])
        out = np.zeros(len(TG))
        if not a: return out
        lg, y = self.logits(self.ctx(r['prompt'], system), a)
        with torch.no_grad():
            out = torch.stack([torch.log_softmax(lg / T, -1).gather(1, y[:, None]).double().sum() for T in TG]).cpu().numpy()
        return out
    def ll_quick(self, r, T=0.7):
        torch = self.torch
        a = self.ans(r['response'], r['m'])
        if not a: return 0.0
        lg, y = self.logits(self.ctx(r['prompt']), a)
        return float(torch.log_softmax(lg / T, -1).gather(1, y[:, None]).double().sum())
    def resolve_date(self, rows):
        """Templates that print today's date (Llama 3.2) must use the date of the answers, not the date of this run."""
        if not self.dynamic_date: return None
        ts = sorted(r['ts'] for r in rows if r.get('ts'))
        cands = []
        for t in ([ts[0], ts[len(ts) // 2], ts[-1]] if ts else [time.time()]):
            for d in (0, -86400, 86400):
                s = time.strftime('%d %b %Y', time.gmtime(t + d))
                if s not in cands: cands.append(s)
        if '26 Jul 2024' not in cands: cands.append('26 Jul 2024')
        rnd = random.Random(0); sample = [r for r in rows if len(r['response']) > 40]; rnd.shuffle(sample); sample = sample[:40]
        best = None
        for s in cands:
            self.date = s; self._ctx = {}
            tot = sum(self.ll_quick(r) for r in sample)
            log(f'   template date {s!r}: log-likelihood {tot:.1f} on {len(sample)} answers')
            if best is None or tot > best[0]: best = (tot, s)
        self.date = best[1]; self._ctx = {}
        return self.date
    def nullsim(self, prompt, m, T, N):
        """N answers sampled from the base at temperature T (plain ancestral sampling, no filter): log-likelihood and text of each"""
        torch = self.torch
        ctx = self.ctx(prompt)
        torch.manual_seed(int(sha(prompt, m, T)[:8], 16))
        bs = 64; lls, texts = [], []; endt = torch.tensor(self.ends, device='cuda')
        body = self.model.base_model; head = self.model.get_output_embeddings()      # logits of the last position only (memory)
        with torch.no_grad():
            while len(lls) < N:
                ids = torch.tensor([ctx], device='cuda').expand(bs, -1)
                o = body(ids, use_cache=True); past = o.past_key_values; lg = head(o.last_hidden_state[:, -1])
                tot = torch.zeros(bs, dtype=torch.float64, device='cuda'); alive = torch.ones(bs, dtype=torch.bool, device='cuda'); seq = []
                for t in range(m):
                    lp = torch.log_softmax(lg.float() / T, -1)
                    tok = torch.multinomial(lp.exp(), 1)
                    tot += torch.where(alive, lp.gather(1, tok)[:, 0].double(), torch.zeros_like(tot)); seq.append(tok)
                    alive &= ~torch.isin(tok[:, 0], endt)
                    if not bool(alive.any()) or t == m - 1: break
                    o = body(tok, past_key_values=past, use_cache=True); past = o.past_key_values; lg = head(o.last_hidden_state[:, -1])
                seqs = torch.cat(seq, 1).tolist(); del past, o, lg
                for b in range(bs):
                    sq = seqs[b]; cut = len(sq)
                    for t, x in enumerate(sq):
                        if x in self.ends: cut = t; break
                    texts.append(self.tok.decode(sq[:cut], skip_special_tokens=True)); lls.append(float(tot[b]))
        torch.cuda.empty_cache()
        return dict(ll=lls[:N], text=texts[:N])
    # ---- fp8 copy, the weight-only path of vLLM "--quantization fp8": float8_e4m3fn weights for every linear layer of the blocks,
    #      one scale (max|W| / 448) per vLLM tensor: q, k, v share one scale, gate and up share one scale, o and down have their own.
    #      Output head, embeddings and norms untouched, activations untouched. (recipe found by the refs lab, exact on ref-02)
    def fp8_prepare(self, recipe='fp8_vllm'):
        """Quantized copy of the linear layers of the blocks. Recipes:
        fp8_vllm (default, exact on the practice fp8 endpoint): e4m3, one scale per merged vLLM tensor;
        fp8_tensor: e4m3, one scale per layer; fp8_channel: e4m3, one scale per output channel;
        int8_channel / int8_tensor: symmetric int8, absmax / 127 per output channel / per layer."""
        torch = self.torch; F8 = torch.float8_e4m3fn
        if getattr(self, '_recipe', None) == recipe: return
        if hasattr(self, '_wb'): self.set_fp8(False)             # back to the base weights before building another copy
        self._lin = [(n, mod) for n, mod in self.model.named_modules() if isinstance(mod, torch.nn.Linear) and 'lm_head' not in n]
        size = sum(mod.weight.numel() * mod.weight.element_size() for _, mod in self._lin)
        self._both_on_gpu = size * 2 < 4e9
        if not hasattr(self, '_wb'):
            self._wb = {n: (mod.weight.data if self._both_on_gpu else mod.weight.data.cpu().pin_memory()) for n, mod in self._lin}
        groups = {}
        for n, mod in self._lin:
            par, _, leaf = n.rpartition('.')
            g = n if recipe != 'fp8_vllm' else par + ('.qkv' if leaf in ('q_proj', 'k_proj', 'v_proj') else '.gu' if leaf in ('gate_proj', 'up_proj') else '.' + leaf)
            groups.setdefault(g, []).append((n, mod))
        self._wq = {}
        for g, mods in groups.items():
            sc = max(max(float(mod.weight.data.float().abs().max()) for _, mod in mods) / 448.0, 1e-12)
            for n, mod in mods:
                w32 = mod.weight.data.float()
                if recipe in ('fp8_vllm', 'fp8_tensor'): q = (w32 / sc).clamp(-448, 448).to(F8).to(torch.float32) * sc
                elif recipe == 'fp8_channel':
                    sr = (w32.abs().amax(1, keepdim=True) / 448.0).clamp_min(1e-12); q = (w32 / sr).clamp(-448, 448).to(F8).to(torch.float32) * sr
                elif recipe == 'int8_channel':
                    sr = (w32.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-12); q = (w32 / sr).round().clamp(-127, 127) * sr
                elif recipe == 'int8_tensor':
                    sr = (w32.abs().max() / 127.0).clamp_min(1e-12); q = (w32 / sr).round().clamp(-127, 127) * sr
                else: raise ValueError(recipe)
                q = q.to(self.dtype)
                self._wq[n] = q if self._both_on_gpu else q.cpu().pin_memory()
                del w32
        self._recipe = recipe
    def set_fp8(self, on):
        if not hasattr(self, '_wb'): self.fp8_prepare()
        src = self._wq if on else self._wb
        for n, mod in self._lin: mod.weight.data = src[n] if self._both_on_gpu else src[n].cuda(non_blocking=True)
    def fp8_chunk(self, rows, recipe='fp8_vllm'):
        """exact likelihood-ratio test of every answer of the chunk between the fp8 copy Q and the base P, on the grid TQ:
        columns = observed log Q - log P, its mean and variance if the text was sampled from P, mean and variance if sampled from Q"""
        import numpy as np
        torch = self.torch
        enc = [(self.ctx(r['prompt']), self.ans(r['response'], r['m'])) for r in rows]
        self.fp8_prepare(recipe)
        self.set_fp8(False); keep = []
        for c, a in enc:                                        # clone: keep only the answer positions, not the logits of the whole context
            if a: lg, y = self.logits(c, a); keep.append((lg.clone(), y)); del lg
            else: keep.append(None)
        self.set_fp8(True); out = []
        with torch.no_grad():
            for (c, a), kb in zip(enc, keep):
                X = np.zeros((len(TQ), 5))
                if a:
                    lgb, y = kb; lgq, _ = self.logits(c, a); acc = []
                    for i, T in enumerate(TQ):
                        lp = torch.log_softmax(lgb / T, -1); lq = torch.log_softmax(lgq / T, -1); d = lq - lp
                        p = lp.exp(); q = lq.exp()
                        m0 = (p * d).sum(-1); v0 = (p * d * d).sum(-1) - m0 * m0
                        m1 = (q * d).sum(-1); v1 = (q * d * d).sum(-1) - m1 * m1
                        acc.append(torch.stack([d.gather(1, y[:, None]).double().sum(), m0.double().sum(), v0.double().sum(), m1.double().sum(), v1.double().sum()]))
                    X = torch.stack(acc).cpu().numpy()
                out.append(X)
        self.set_fp8(False); del keep; torch.cuda.empty_cache()
        return out

# ----------------------------------------------------------------------------------------------------------- statistics
def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def fisher_z(ps):
    if not ps: return 0.0
    x = -2 * sum(math.log(max(p, 1e-300)) for p in ps); k = len(ps)
    return (x - 2 * k) / math.sqrt(4 * k)

def tgi(T):
    return min(range(len(TG)), key=lambda i: abs(TG[i] - T))

def calib(fs, T0=None):
    """fs: list of per-answer features. Returns MLE temperature and calibration statistics."""
    import numpy as np
    fs = [f for f in fs if f['n'] > 0]
    if not fs: return None
    S = sum(f['S'] for f in fs); n = sum(f['n'] for f in fs)
    ib = int(np.argmax(S[:, 0])); T = TG[ib]; se = float('nan')
    if 0 < ib < len(TG) - 1:
        y0, y1, y2 = S[ib - 1, 0], S[ib, 0], S[ib + 1, 0]; den = y0 - 2 * y1 + y2
        if den < 0:
            T = TG[ib] + 0.5 * (y0 - y2) / den * 0.02; se = math.sqrt(-(0.02 ** 2) / den)
    def z(i): return float((S[i, 0] + S[i, 1]) / math.sqrt(max(S[i, 2], 1e-9)))
    out = dict(n_ans=len(fs), tokens=int(n), T=round(float(T), 3), T_se=round(se, 4), zT=round(z(ib), 2),
               disp=round(float((S[ib, 4] - S[ib, 2]) / math.sqrt(max(S[ib, 3], 1e-9))), 2))
    per = [float((f['S'][ib, 0] + f['S'][ib, 1]) / math.sqrt(max(f['S'][ib, 2], 1e-9))) for f in fs]
    mu = sum(per) / len(per); out['ans_sd'] = round((sum((x - mu) ** 2 for x in per) / max(1, len(per) - 1)) ** 0.5, 2)
    # stop test: answers that ended before max_tokens against the end-of-turn probability summed along the text (at the fitted temperature)
    out['stops'] = int(sum(f['stop'] for f in fs)); out['stops_expected'] = round(float(S[ib, 6]), 1)
    out['z_stop'] = round(float((out['stops'] - S[ib, 6]) / math.sqrt(max(S[ib, 7], 1e-9))), 2)
    if T0 is not None:
        i0 = tgi(T0); out['z0'] = round(z(i0), 2)
        e1 = S[i0, 5] / n; r1 = sum(f['top1'] for f in fs) / n
        out['top1_z0'] = round(float((r1 - e1) / math.sqrt(max(e1 * (1 - e1) / n, 1e-12))), 2)
    return out

def mart_z(fs, T):
    fs = [f for f in fs if f['n'] > 0]
    if not fs: return 0.0
    i = tgi(T); S = sum(f['S'] for f in fs)
    return float((S[i, 0] + S[i, 1]) / math.sqrt(max(S[i, 2], 1e-9)))

def snap(T):
    return min(1.2, max(0.5, round(T * 10) / 10))

def rep_pvalue(answers, null_texts, R=2000, seed=0):
    """Longest common prefix among the k answers to one prompt, against k simulated base answers:
    returns (observed, p = P(simulated >= observed), excess = (observed - mean) / sd of the simulated values, largest simulated value)."""
    k = len(answers); obs = max(lcp(a, b) for a, b in itertools.combinations(answers, 2))
    rnd = random.Random(seed); ge = 0; vals = []
    for _ in range(R):
        s = [null_texts[rnd.randrange(len(null_texts))] for _ in range(k)]
        v = max(lcp(a, b) for a, b in itertools.combinations(s, 2)); vals.append(v)
        if v >= obs: ge += 1
    mu = sum(vals) / R; sd = (sum((v - mu) ** 2 for v in vals) / R) ** 0.5
    return obs, (1 + ge) / (1 + R), (obs - mu) / max(sd, 0.5), max(vals)

# ------------------------------------------------------------------------------------------------------- stage a: family
def stage_family(rows, cfg, cache):
    """Every candidate scores a sample of each endpoint's answers; an endpoint joins the candidate that explains its text best."""
    import numpy as np
    eps = sorted({r['endpoint'] for r in rows})
    sample = {}
    for e in eps:
        rs = sorted((r for r in rows if r['endpoint'] == e), key=lambda r: (-len(r['response']), r['id']))
        sample[e] = rs[:cfg.get('family_answers', 8)]
    score = {e: {} for e in eps}; dates = {}
    for repo in cfg['candidates']:
        F = cache.get('feat_' + slug(repo)); meta = cache.get('meta_' + slug(repo))
        need_model = True; base = None
        def get_base():
            nonlocal base
            if base is None: base = Base(repo, dates.get(repo))
            return base
        # templates that print today's date: provisional date = median day of the answers (refined per family in measure_family)
        if 'dynamic' not in meta:
            from transformers import AutoTokenizer
            meta['dynamic'] = 'strftime_now' in (AutoTokenizer.from_pretrained(repo).chat_template or ''); cache.save('meta_' + slug(repo))
        date = None
        if meta['dynamic']:
            ts = sorted(r['ts'] for r in rows if r.get('ts'))
            date = time.strftime('%d %b %Y', time.gmtime(ts[len(ts) // 2] if ts else time.time()))
        dates[repo] = date
        for e in eps:
            tot = np.zeros(len(TG)); chars = 0
            for r in sample[e]:
                k = sha(repo, date, r['m'], r['prompt'], r['response'], FEAT)
                if k not in F: F[k] = get_base().feats(r)
                tot += F[k]['S'][:, 0]; chars += max(1, len(r['response']))
            score[e][repo] = float(tot.max()) / chars * 1000
        cache.save('feat_' + slug(repo))
        if base is not None: base.free(); del base
        log(f'stage a: {repo} scored; best for', sum(1 for e in eps if max(score[e], key=score[e].get) == repo), 'endpoints so far')
    fam = {}
    for e in eps:
        o = sorted(score[e].items(), key=lambda kv: -kv[1])
        fam[e] = dict(base=o[0][0], ll_per_kchar=round(o[0][1], 1), margin_per_kchar=round(o[0][1] - o[1][1], 1) if len(o) > 1 else None,
                      scores={k: round(v, 1) for k, v in o})
    return fam, dates

# ------------------------------------------------------------------------------------------------ stages b, c, d: measure
def measure_family(repo, date, rows, cfg, cache):
    import numpy as np
    thr = cfg['thr']
    eps = sorted({r['endpoint'] for r in rows})
    F = cache.get('feat_' + slug(repo)); NUL = cache.get('null_' + slug(repo)); meta = cache.get('meta_' + slug(repo))
    base = None
    if meta.get('dynamic'):
        dk = 'date_' + sha(*eps)[:12]
        if dk not in meta:
            base = Base(repo, date); meta[dk] = base.resolve_date(rows); cache.save('meta_' + slug(repo))
        date = meta[dk]
        if base is not None: base.date = date; base._ctx = {}
        log(f'   template date: {date}')
    def get_base():
        nonlocal base
        if base is None: base = Base(repo, date)
        return base
    def K(r, tag=FEAT): return sha(repo, date, r['m'], r['prompt'], r['response'], tag)
    miss = [r for r in rows if K(r) not in F]
    if miss:
        log(f'   base features: {len(miss)} answers to score')
        for i, r in enumerate(miss):
            F[K(r)] = get_base().feats(r)
            if i % 500 == 499: cache.save('feat_' + slug(repo))
        cache.save('feat_' + slug(repo))
    for r in rows: r['f'] = F[K(r)]
    # plain-context answers under alternative system prompts (is the default system prompt still there?)
    alts = cfg.get('alt_systems', ['You are a helpful assistant.', ''])
    miss = [(r, s) for r in rows if not r['reset'] for s in alts if K(r, 'alt:' + s) not in F]
    if miss:
        log(f'   alternative system prompts: {len(miss)} scorings')
        for r, s in miss: F[K(r, 'alt:' + s)] = get_base().ly_grid(r, s)
        cache.save('feat_' + slug(repo))
    miss = [r for r in rows if K(r, 'eot') not in F]
    if miss:
        log(f'   end-of-answer check: {len(miss)} answers')
        for r in miss: F[K(r, 'eot')] = get_base().eot_info(r)
        cache.save('feat_' + slug(repo))
    nc = collections.Counter()
    for r in rows:
        x = F[K(r, 'eot')]
        r['nc'] = bool(x['n_raw'] > r['m'] or (x['lp07'] is not None and x['lp07'] < thr['eot_lp']))
        nc[r['endpoint']] += r['nc']
    if sum(nc.values()): log(f'   answers left out (tokenization not canonical or impossible end of turn): {dict(nc)}')
    all_rows = rows; rows = [r for r in rows if not r['nc']]
    miss = [r for r in rows if K(r, 'trunc') not in F]
    if miss:
        log(f'   truncated-sampling counts: {len(miss)} answers')
        for r in miss: F[K(r, 'trunc')] = get_base().trunc(r)
        cache.save('feat_' + slug(repo))
    by = collections.defaultdict(list)
    for r in rows: by[r['endpoint']].append(r)
    # ---- c. temperature
    st = {e: {} for e in eps}
    for e in eps:
        st[e]['n_answers'] = len(by[e]); st[e]['n_left_out'] = int(nc[e])
        c = calib([r['f'] for r in by[e] if r['long'] and r['f']['n'] >= thr['min_tok']])
        st[e]['cal'] = c
    Ts = sorted(st[e]['cal']['T'] for e in eps if st[e]['cal'])
    T0 = round(round(Ts[max(0, (len(Ts) - 1) // 4)] / 0.02) * 0.02, 2) if Ts else 0.7      # family temperature = lower quartile of the fitted temperatures
    for e in eps:
        st[e]['cal'] = calib([r['f'] for r in by[e] if r['long'] and r['f']['n'] >= thr['min_tok']], T0)
        ca = calib([r['f'] for r in by[e] if r['f']['n'] >= 4])
        st[e]['T_all'] = ca['T'] if ca else None                     # maximum-likelihood temperature over every answer (used by the fingerprint)
        # truncated sampling (top-k / top-p): tokens beyond rank k or nucleus p, observed against expected at the fitted temperature
        Te = st[e]['cal']['T'] if st[e]['cal'] else T0; it = tgi(Te)
        sel = [r for r in by[e] if r['reset'] and r['f']['n'] >= 4]
        if len(sel) < 5: sel = [r for r in by[e] if r['f']['n'] >= 4]
        tr = dict(tokens=int(sum(r['f']['n'] for r in sel)), z=0.0, worst=None, counts={})
        if sel:
            ok = np.sum([F[K(r, 'trunc')]['obs_k'] for r in sel], 0); X = np.sum([F[K(r, 'trunc')]['X'][it] for r in sel], 0)
            tests = [(f'k{k}', float(ok[j]), float(X[j])) for j, k in enumerate(KS)] + \
                    [(f'p{q}', float(X[len(KS) + j]), float(X[len(KS) + len(PS) + j])) for j, q in enumerate(PS)]
            for name, o, x in tests:
                tr['counts'][name] = [int(o), round(x, 1)]
                if x >= thr['trunc_min_exp']:
                    z = (o - x) / math.sqrt(x)
                    if z < tr['z']: tr['z'] = round(z, 2); tr['worst'] = f'{name}: {int(o)} observed for {x:.0f} expected'
                    if o <= thr['trunc_ratio'] * x: tr['detected'] = True
        st[e]['trunc'] = tr
    log(f'   family temperature T0 = {T0}')
    # ---- b. prompt disguises and c. repeatability: simulated base null
    T0s = snap(T0)
    plain_prompts = collections.Counter(); rep_prompts = collections.Counter()
    for r in rows:
        if not r['reset'] and r['m'] <= 128: plain_prompts[(r['prompt'], r['m'])] += 10 if r['probe'] else 1
        if r['k'] >= 2: rep_prompts[(r['prompt'], r['m'])] = max(rep_prompts[(r['prompt'], r['m'])], r['k'])
    probes = [k for k, v in plain_prompts.items() if any(r['probe'] and (r['prompt'], r['m']) == k for r in rows)]
    order = probes + [k for k, _ in sorted(rep_prompts.items(), key=lambda kv: -kv[1]) if k not in probes]
    order += [k for k, _ in plain_prompts.most_common() if k not in order]
    order = order[:thr['max_null_prompts']]
    own = {e: snap(st[e]['cal']['T']) if st[e]['cal'] else T0s for e in eps}
    need = set()
    for (p, m) in order:
        need.add((p, m, T0s))
        if (p, m) not in probes: continue                     # own-temperature null only for the extraction prompts (cost)
        for e in eps:
            if own[e] != T0s and any((r['prompt'], r['m']) == (p, m) and not r['reset'] for r in by[e]): need.add((p, m, own[e]))
    todo = [k for k in sorted(need, key=lambda k: (k[2], k[1])) if sha(repo, date, *k, thr['null_n']) not in NUL]
    if todo:
        log(f'   base null: {len(todo)} prompt/temperature simulations of {thr["null_n"]} answers')
        for i, (p, m, T) in enumerate(todo):
            NUL[sha(repo, date, p, m, T, thr['null_n'])] = get_base().nullsim(p, m, T, thr['null_n'])
            if i % 10 == 9: cache.save('null_' + slug(repo))
        cache.save('null_' + slug(repo))
    def null(p, m, T): return NUL.get(sha(repo, date, p, m, T, thr['null_n']))
    for e in eps:
        Te = st[e]['cal']['T'] if st[e]['cal'] else T0
        plain = [r for r in by[e] if not r['reset']]
        d = dict(n_plain=len(plain), z_plain_own=round(mart_z([r['f'] for r in plain], Te), 2), z_plain_T0=round(mart_z([r['f'] for r in plain], T0), 2))
        for tag, T in (('own', own[e]), ('T0', T0s)):
            ps, pp, worst = [], [], []
            for r in plain:
                nu = null(r['prompt'], r['m'], T)
                if nu is None: continue
                ll = float(r['f']['S'][tgi(T), 0]) if r['f']['n'] else 0.0
                p = (1 + sum(x <= ll + 0.05 + 0.01 * abs(ll) for x in nu['ll'])) / (len(nu['ll']) + 1)
                ps.append(p); worst.append((p, r['id']))
                if r['probe']: pp.append(p)
            d[f'short_fz_{tag}'] = round(fisher_z(ps), 2); d[f'probe_fz_{tag}'] = round(fisher_z(pp), 2)
            d[f'n_short_{tag}'] = len(ps); d[f'n_probe_{tag}'] = len(pp); d[f'short_p02_{tag}'] = sum(p < 0.02 for p in ps)
            d[f'worst_{tag}'] = [i for _, i in sorted(worst)[:3]]
        d['T_null_own'] = own[e]
        # likelihood ratio: best alternative system prompt against the default one, on the plain-context answers at the endpoint's own temperature
        it = tgi(Te); best = None
        for s in alts:
            dif = [(float(F[K(r, 'alt:' + s)][it] - r['f']['S'][it, 0]), r['probe']) for r in plain if r['f']['n'] > 0]
            tot = sum(x for x, _ in dif)
            if best is None or tot > best[0]: best = (tot, sum(x for x, p in dif if p), sum(x > 1 for x, _ in dif), sum(x < -1 for x, _ in dif), s)
        if best: d.update(alt_llr=round(best[0], 2), alt_llr_probe=round(best[1], 2), alt_n_for=best[2], alt_n_against=best[3], alt_system=best[4])
        st[e]['plain'] = d
        # repeatability
        groups = collections.defaultdict(list)
        for r in by[e]:
            if r['k'] >= 2: groups[(r['prompt'], r['m'])].append(r['response'])
        ps, mx, kmax, n_rep, best = [], 0, 0, 0, None
        for (p, m), ans in groups.items():
            if len(ans) < 2: continue                                 # the other answers of the group were left out
            raw = max(lcp(a, b) for a, b in itertools.combinations(ans, 2)); nu = null(p, m, T0s)
            if len(ans) >= thr['rep_min']: mx = max(mx, raw); kmax = max(kmax, len(ans)); n_rep += len(ans)
            if nu is not None:
                obs, pv, ex, top = rep_pvalue(ans, nu['text'], seed=int(sha(e, p)[:6], 16)); ps.append(pv)
                if best is None or ex > best['excess']: best = dict(excess=round(ex, 1), prefix=obs, largest_simulated=top, p=round(pv, 5), k=len(ans))
        # watermark signature: one prompt whose answers share a prefix longer than every one of the 2000 simulated draws, far outside their spread
        st[e]['rep'] = dict(groups=len(groups), groups_tested=len(ps), n_repeat_answers=n_rep, max_k=kmax, max_lcp=mx, fz=round(fisher_z(ps), 2),
                            min_p=round(min(ps), 5) if ps else None, best=best,
                            excess=best['excess'] if best else 0.0, beyond=bool(best and best['prefix'] > best['largest_simulated']))
    # ---- d. fp8 likelihood ratio, calibrated: z0 ~ N(0,1) if the text comes from the base, z1 ~ N(0,1) if it comes from the quantized copy
    def tqi(T): return min(range(len(TQ)), key=lambda i: abs(TQ[i] - T))
    def quant_pass(recipe, tag):
        miss = [r for r in rows if K(r, tag) not in F]
        if miss:
            log(f'   quantized copy {recipe}: {len(miss)} answers to score')
            b = get_base(); V = b.model.get_output_embeddings().weight.shape[0]; chunk, size = [], 0
            def flush():
                nonlocal chunk, size
                if chunk:
                    for r, X in zip(chunk, b.fp8_chunk(chunk, recipe)): F[K(r, tag)] = X
                chunk, size = [], 0
            for r in miss:
                n = min(r['m'], len(r['response']) + 1); chunk.append(r); size += n * V * 4
                if size > 0.8e9: flush()
            flush(); cache.save('feat_' + slug(repo))
        res = {}
        for e in eps:
            Te = st[e]['cal']['T'] if st[e]['cal'] else T0; res[e] = {}
            for name, sel, T in (('all', by[e], T0), ('reset', [r for r in by[e] if r['reset']], Te), ('all_own', by[e], Te)):
                xs = [(F[K(r, tag)][tqi(T)], r['f']['n']) for r in sel if r['f']['n'] > 0]
                if not xs: continue
                S = sum(x for x, _ in xs); n = sum(k for _, k in xs); s = [float(x[0]) for x, _ in xs]; mu = sum(s) / len(s)
                sd = (sum((x - mu) ** 2 for x in s) / max(1, len(s) - 1)) ** 0.5
                res[e][name] = dict(
                    llr_per_ktok=round(1000 * float(S[0]) / n, 2), z=round(mu / (sd / math.sqrt(len(s)) + 1e-12), 2), n_ans=len(s), tokens=int(n),
                    exp_base_per_ktok=round(1000 * float(S[1]) / n, 2), exp_fp8_per_ktok=round(1000 * float(S[3]) / n, 2),
                    z0=round(float((S[0] - S[1]) / math.sqrt(max(S[2], 1e-9))), 2), z1=round(float((S[0] - S[3]) / math.sqrt(max(S[4], 1e-9))), 2),
                    frac=round(float((S[0] - S[1]) / max(S[3] - S[1], 1e-9)), 3))
        return res
    def hit(v): return bool(v.get('reset') and v['reset']['z0'] >= thr['fp8_z0'] and v['reset']['frac'] >= thr['fp8_frac'])
    main = quant_pass('fp8_vllm', 'fp8m')
    for e in eps: st[e]['fp8'] = main[e]; st[e]['fp8']['recipe'] = 'fp8_vllm'
    if not any(hit(main[e]) for e in eps):
        # no endpoint of the family carries the main fp8 recipe: try the other quantization recipes before concluding
        for recipe in cfg.get('quant_recipes', ['fp8_tensor', 'fp8_channel', 'int8_channel']):
            alt = quant_pass(recipe, 'q:' + recipe)
            for e in eps:
                st[e].setdefault('quant', {})[recipe] = alt[e].get('reset')
                if hit(alt[e]) and not hit(st[e]['fp8']): st[e]['fp8'] = dict(alt[e], recipe=recipe)
            log(f'   quantized copy {recipe}: ' + ' '.join(f"{e[-2:]}:{(alt[e].get('reset') or {}).get('z0')}/{(alt[e].get('reset') or {}).get('frac')}" for e in eps))
    if base is not None: base.free()
    for r in all_rows: r.pop('f', None)
    return st, T0, date

# ------------------------------------------------------------------------------------------------ stage e: adapter proxy
class Proxy:
    """LoRA proxy learned from answers (rank 16, all linear layers); dose s scales the adapter, temperature is fitted jointly."""
    def __init__(self, repo, date, thr):
        self.repo, self.date, self.thr = repo, date, thr
        self.b = Base(repo, date, train=True)
    def enc(self, r):
        return (self.b.ctx(r['prompt']), self.b.ans(r['response'], r['m']))
    def _wrap(self):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=self.thr['rank'], lora_alpha=2 * self.thr['rank'], target_modules='all-linear', lora_dropout=0.0, bias='none')
        self.m = get_peft_model(self.b.model, cfg); self._s0 = {}
    def train(self, examples, seed=0, T=1.0):
        """examples: (context ids, answer ids). The loss is taken at temperature T (the fitted temperature of the endpoint when
        train_own_T is on: the adapter then learns the shift of the weights, not the sharpening of the sampler).
        Returns the adapter weights (CPU) so the proxy can be reloaded for the cross tests."""
        import torch
        torch.manual_seed(seed); rnd = random.Random(seed)
        self._wrap(); self.m.train()
        opt = torch.optim.AdamW([p for p in self.m.parameters() if p.requires_grad], lr=self.thr['lr'])
        ex = [x for x in examples if len(x[1]) >= 4]
        for ep in range(self.thr['epochs']):
            rnd.shuffle(ex); tot = 0.0; n = 0
            for x in ex:
                ctx, a = x[0], x[1]; Tx = x[2] if len(x) > 2 else T
                ids = torch.tensor([ctx + a], device='cuda')
                lg = self.m(ids).logits[0, len(ctx) - 1:-1].float()
                loss = torch.nn.functional.cross_entropy(lg / Tx, ids[0, len(ctx):])
                loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss) * len(a); n += len(a)
            log(f'      epoch {ep}: train nll/token {tot / max(1, n):.3f} on {len(ex)} answers')
        self.m.eval()
        return {k: v.detach().cpu().clone() for k, v in self.m.state_dict().items() if 'lora_' in k}
    def load(self, sd):
        self._wrap(); self.m.load_state_dict(sd, strict=False); self.m.eval()
    def set_scale(self, s):
        for i, mod in enumerate(self.m.modules()):
            if hasattr(mod, 'scaling') and isinstance(mod.scaling, dict) and 'default' in mod.scaling:
                if i not in self._s0: self._s0[i] = mod.scaling['default']
                mod.scaling['default'] = self._s0[i] * s
    def score(self, examples, sg=SG):
        """[n_answers, len(sg), len(DT)] summed log-probs"""
        import numpy as np, torch
        out = np.zeros((len(examples), len(sg), len(DT)))
        with torch.no_grad():
            for si, s in enumerate(sg):
                self.set_scale(s)
                for ai, (ctx, a) in enumerate(examples):
                    ids = torch.tensor([ctx + a], device='cuda')
                    lg = self.m(ids).logits[0, len(ctx) - 1:-1].float(); y = ids[0, len(ctx):]
                    for ti, T in enumerate(DT):
                        out[ai, si, ti] = float(torch.log_softmax(lg / T, -1).gather(1, y[:, None]).double().sum())
        self.set_scale(1.0)
        return out
    def unload(self):
        self.b.model = self.m.unload(); del self.m
        self.b.model.eval()
    def free(self):
        self.b.free()

def fit_dose(X, ntok, sg=SG):
    """X [n, S, T] -> best dose and temperature, gain over dose 0 (the lead's statistic), cross-fitted gain and z, bootstrap of the dose"""
    import numpy as np
    n = X.shape[0]
    if n == 0: return None
    tot = X.sum(0); si, ti = np.unravel_index(int(np.argmax(tot)), tot.shape); t0 = int(np.argmax(tot[0]))
    gain = float(tot[si, ti] - tot[0, t0])
    prof = tot.max(1); s_hat = float(sg[si])
    if 0 < si < len(sg) - 1:                                   # continuous dose: parabola through the three grid points around the best one
        c = np.polyfit(sg[si - 1:si + 2], prof[si - 1:si + 2], 2)
        if c[0] < 0: s_hat = float(min(sg[si + 1], max(sg[si - 1], -c[1] / (2 * c[0]))))
    idx = np.arange(n); half = [idx[idx % 2 == 0], idx[idx % 2 == 1]]; cf = np.zeros(n)
    if n >= 4:                                                 # cross-fit: dose and temperatures chosen on one half of the answers, gain read on the other
        for h in (0, 1):
            tr = X[half[h]].sum(0); a, b = np.unravel_index(int(np.argmax(tr)), tr.shape); c = int(np.argmax(tr[0]))
            cf[half[1 - h]] = X[half[1 - h], a, b] - X[half[1 - h], 0, c]
    sd = float(cf.std(ddof=1)) if n > 1 else 0.0
    rng = np.random.default_rng(1); boots = []
    for _ in range(300):
        t = X[rng.integers(0, n, n)].sum(0); boots.append(sg[int(np.argmax(t)) // t.shape[1]])
    boots.sort()
    return dict(n=int(n), s=sg[si], s_hat=round(s_hat, 3), T=DT[ti], T_s0=DT[t0], gain=round(gain, 2), gain_per_answer=round(gain / n, 3),
                gain_per_ktok=round(1000 * gain / max(1, ntok), 3), cf_gain_per_answer=round(float(cf.mean()), 3),
                cf_gain_per_ktok=round(1000 * float(cf.sum()) / max(1, ntok), 3),
                cf_z=round(float(cf.mean() / (sd / math.sqrt(n) + 1e-12)), 2), s_q=[boots[30], boots[150], boots[270]])

def fold_of(prompt):
    return int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 2

def self_ok(g, thr):
    return bool(g and g['cf_gain_per_answer'] >= thr['self_gain'] and g['cf_z'] >= thr['lora_z'])

def pool_ok(g, thr):
    return bool(g and g['cf_gain_per_answer'] >= thr['lora_gain'] and g['cf_z'] >= thr['lora_z'])

def stage_lora(repo, date, rows, st, pre, cfg, do_cross=True):
    """1. self test of every hot endpoint that no disguise explains (proxy trained on half of its prompts, gain on the other half);
       2. pooled proxy on the strongest confirmed endpoints, out-of-sample gain for every other endpoint;
       3. cross test between adapter traps (proxy of A alone, dose on B relative to the held-out dose on A)."""
    import numpy as np
    thr = cfg['thr']
    eps = sorted({r['endpoint'] for r in rows})
    px = Proxy(repo, date, thr)
    data = {e: [[], []] for e in eps}; ntok = collections.Counter()
    for r in rows:
        if not r['long']: continue
        ctx, a = px.enc(r)
        if len(a) >= 4: data[r['endpoint']][fold_of(r['prompt'])].append((ctx, a)); ntok[r['endpoint']] += len(a)
    out = dict(self={}, pooled={}, cross={}, train=[], confirmed=[])
    adapters = {}
    Ttr = {e: (float(st[e]['cal']['T']) if thr['train_own_T'] and st[e]['cal'] else 1.0) for e in eps}
    def self_test(e):
        parts = []
        for f in (0, 1):
            if len(data[e][f]) < 3 or len(data[e][1 - f]) < 3: continue
            adapters[(e, f)] = px.train(list(data[e][f]), seed=f, T=Ttr[e]); parts.append(px.score(data[e][1 - f])); px.unload()
        if parts:
            out['self'][e] = fit_dose(np.concatenate(parts, 0), ntok[e]); log(f'      self {e}: {out["self"][e]}')
    cand = [e for e in eps if (pre[e]['hot'] or pre[e]['cold']) and not pre[e]['explained']]
    log(f'   stage e: self test on {len(cand)} hot endpoints that no disguise explains: {" ".join(cand)}')
    for e in cand: self_test(e)
    confirmed = sorted((e for e in cand if self_ok(out['self'].get(e), thr)), key=lambda e: -out['self'][e]['cf_gain_per_answer'])
    train = confirmed[:thr['n_train']]
    out['train'] = train; out['confirmed'] = confirmed
    if train:
        log(f'   stage e: pooled proxy trained on {" ".join(train)}')
        px.train([(c, a, Ttr[e]) for e in train for f in (0, 1) for c, a in data[e][f]], seed=7)
        for e in eps:
            ex = data[e][0] + data[e][1]
            if ex: out['pooled'][e] = fit_dose(px.score(ex), ntok[e])
        px.unload()
        for e in eps:
            if e in out['pooled']: log(f'      pooled {e}: {out["pooled"][e]}')
    weak = [e for e in eps if e not in confirmed and pool_ok(out['pooled'].get(e), thr) and not pre[e]['watermark'] and not pre[e]['fp8']]
    traps = (confirmed + weak)[:8]
    if do_cross and len(traps) >= 2:
        log(f'   stage e: cross tests between {" ".join(traps)}')
        for a in traps:
            if a not in out['self']: self_test(a)
        for a in traps:
            parts = collections.defaultdict(list)
            for f in (0, 1):
                if (a, f) not in adapters: continue
                px.load(adapters[(a, f)])
                for b in traps:
                    if b != a and data[b][1 - f]: parts[b].append(px.score(data[b][1 - f]))
                px.unload()
            sa = (out['self'].get(a) or {}).get('s_hat') or 0.0
            for b, ps in parts.items():
                d = fit_dose(np.concatenate(ps, 0), ntok[b]); d['self_dose_of_trainer'] = sa
                d['ratio'] = round(d['s_hat'] / sa, 2) if sa > 0 else None
                out['cross'][f'{a}>{b}'] = d; log(f'      cross {a}>{b}: dose {d["s_hat"]} T {d["T"]} ratio {d["ratio"]} gain {d["cf_gain_per_answer"]} z {d["cf_z"]}')
    px.free()
    return out

# ------------------------------------------------------------------------------------------- stage e: gradient fingerprint
def gfp_stats(eps, K, G, nt):
    """Self test, pair test, direction cosine and dose ratio from the Gram matrix of the (endpoint, fold) gradient sums.
    z = sum / sqrt(sum of squares) over the cross-fold terms (gradfp.py). The pair z assumes that BOTH endpoints have mean-zero
    gradients, so it is inflated when the partner is a real trap. The calibrated reading (gradcal.py, simulated bench) is p(a | b):
        r_k = g_a,k . (sum of the folds of b other than k), independent over k and mean zero if a is the base model, whatever b is;
        p(a | b) = one-sided sign-flip p-value over the folds of a (exact up to 13 folds). It tests a, not b."""
    import numpy as np
    E = len(eps); off = ~np.eye(K, dtype=bool); up = np.triu(np.ones((K, K), dtype=bool), 1)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=K))) if K <= 13 else np.random.default_rng(0).choice([-1.0, 1.0], size=(20000, K))
    def blk(a, b): return G[a * K:(a + 1) * K, b * K:(b + 1) * K]
    def n_(a): return nt[a * K:(a + 1) * K]
    def tables(drop=None):
        Z = np.zeros((E, E)); F = np.zeros((E, E))
        for a in range(E):
            for b in range(E):
                mask = (off if a != b else up).copy()
                if drop is not None: mask[drop, :] = False; mask[:, drop] = False
                v = blk(a, b)[mask]; w = np.outer(n_(a), n_(b))[mask]
                Z[a, b] = v.sum() / max(math.sqrt((v ** 2).sum()), 1e-30); F[a, b] = v.sum() / max(w.sum(), 1.0)
        return Z, F
    Z, F = tables(); jk = [tables(d)[1] for d in range(K)]
    def cos(Fm, a, b): return float(Fm[a, b] / math.sqrt(Fm[a, a] * Fm[b, b])) if Fm[a, a] > 0 and Fm[b, b] > 0 else None
    def ratio(Fm, a, b): return float(Fm[a, b] / Fm[b, b]) if Fm[b, b] > 0 else None       # dose of a in units of the dose of b
    def jse(fn, a, b):
        xs = [fn(Fd, a, b) for Fd in jk]; xs = [x for x in xs if x is not None]
        if len(xs) < K: return None
        mu = sum(xs) / len(xs); return round(math.sqrt((len(xs) - 1) / len(xs) * sum((x - mu) ** 2 for x in xs)), 3)
    def along(a, b):
        r = (blk(a, b) * off).sum(1); ok = n_(a) > 0; r = r[ok]
        if len(r) < 4: return None, None
        t = float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)) + 1e-30)); null = signs[:, ok] @ r
        return round(t, 2), float((1 + (null >= r.sum() - 1e-12).sum()) / (1 + len(null)))
    def pself(a):
        M = blk(a, a) * off; null = np.einsum('rk,kl,rl->r', signs, M, signs) / 2
        return float((1 + (null >= M.sum() / 2 - 1e-12).sum()) / (1 + len(null)))
    out = dict(self={}, pair={}, p_min=1.0 / (1 + len(signs)))
    for a, e in enumerate(eps):
        out['self'][e] = dict(z=round(float(Z[a, a]), 2), p=round(pself(a), 5), effect=round(float(F[a, a]) * 1e6, 1), tokens=int(n_(a).sum()),
                              folds=int((n_(a) > 0).sum()))
        out['pair'][e] = {}
        for b, o in enumerate(eps):
            if a == b: continue
            c = cos(F, a, b); r = ratio(F, a, b); t, p = along(a, b)
            out['pair'][e][o] = dict(z=round(float(Z[a, b]), 2), t=t, p=None if p is None else round(p, 5), cos=None if c is None else round(c, 3),
                                     ratio=None if r is None else round(r, 3), cos_se=jse(cos, a, b), ratio_se=jse(ratio, a, b))
    return out

def stage_gradfp(repo, date, rows, st, T0, cfg, cache=None):
    """Gradient fingerprint (score test, the lead's gradfp.py): for every answer, gradient of its log-likelihood under the base copy, at the
    endpoint's own fitted temperature, with respect to a random low-rank sketch of every linear weight (LoRA B at 0 with random A).
    If the endpoint is the base model, whatever its temperature, these gradients have mean zero and sums over disjoint prompt folds are
    uncorrelated."""
    import numpy as np, torch
    from peft import LoraConfig, get_peft_model
    thr = cfg['thr']; K = int(thr['gfp_folds']); R = int(thr['gfp_rank'])
    eps = sorted({r['endpoint'] for r in rows}); E = len(eps)
    # the result is cached under a key made of every answer and every temperature that enters it
    sig = sha(repo, date, K, R, thr['gfp_seed'], *sorted(sha(r['endpoint'], r['m'], r['prompt'], r['response'], r['long']) for r in rows if not r.get('nc')),
              *[f"{e}:{(st[e]['cal'] or {}).get('T')}:{st[e].get('T_all')}" for e in eps])
    GC = cache.get('gfp_' + slug(repo)) if cache else {}
    def stats(c):
        out = dict(all=gfp_stats(eps, K, c['G_all'], c['nt_all']))
        for e in eps:
            log(f"      {e}: self z {out['all']['self'][e]['z']:+6.2f} (p {out['all']['self'][e]['p']:.4f}) | calibrated p(row | column) x1000: " +
                ' '.join(f"{o[-2:]}:{1000 * out['all']['pair'][e][o]['p']:.0f}" if out['all']['pair'][e][o]['p'] is not None else f"{o[-2:]}:-" for o in eps if o != e))
        return out
    if sig in GC and 'G_all' in GC[sig]:
        log('   stage e: gradient fingerprint read from the cache'); return stats(GC[sig])
    b = Base(repo, date, train=True)
    torch.manual_seed(int(thr['gfp_seed']))
    model = get_peft_model(b.model, LoraConfig(r=R, lora_alpha=R, target_modules='all-linear', lora_dropout=0.0, bias='none')); model.eval()
    Bs = [p for n, p in model.named_parameters() if 'lora_B' in n]
    for n, p in model.named_parameters(): p.requires_grad_('lora_B' in n)
    D = sum(p.numel() for p in Bs)
    log(f'   stage e: gradient fingerprint, sketch dimension {D}, rank {R}, {K} folds')
    # the gradient is taken at the maximum-likelihood temperature of the very answers that are summed (every usable answer of the
    # endpoint), so the temperature direction is removed exactly
    n = E * K
    S = torch.zeros(n, D, dtype=torch.float32)                    # (endpoint, fold) sums, CPU
    nt = np.zeros(n)
    seen = set()
    for ei, e in enumerate(eps):
        Tl = float(st[e]['cal']['T']) if st[e]['cal'] else T0
        Ta = float(st[e].get('T_all') or Tl)
        on_gpu = K * D * 4 < 1.5e9                                 # fold sums of this endpoint on the GPU when they are small
        buf = torch.zeros(K, D, dtype=torch.float32, device='cuda' if on_gpu else 'cpu')
        for r in rows:
            if r['endpoint'] != e or r.get('nc') or (e, r['prompt'], r['response']) in seen: continue
            seen.add((e, r['prompt'], r['response']))               # an identical answer to an identical request counts once
            ctx, a = b.ctx(r['prompt']), b.ans(r['response'], r['m'])
            if len(a) < 4: continue
            f = int(hashlib.md5(r['prompt'].encode()).hexdigest(), 16) % K
            ids = torch.tensor([ctx + a], device='cuda')
            lg = model(ids).logits[0, len(ctx) - 1:-1].float(); y = ids[0, len(ctx):]
            ll = torch.log_softmax(lg / Ta, -1).gather(1, y[:, None]).sum()
            grads = torch.autograd.grad(ll, Bs)
            buf[f] += torch.cat([g.reshape(-1).float() for g in grads]).to(buf.device); nt[ei * K + f] += len(a)
            del lg, ll, grads
        S[ei * K:(ei + 1) * K] = buf.cpu(); del buf
    del model, Bs; b.free()
    G = torch.zeros(n, n, dtype=torch.float64)
    for c in range(0, D, 1_000_000):
        x = S[:, c:c + 1_000_000].cuda().double(); G += (x @ x.T).cpu(); del x
    del S; torch.cuda.empty_cache()
    c = dict(G_all=G.numpy(), nt_all=nt)
    if cache is not None:
        GC[sig] = c; cache.save('gfp_' + slug(repo))
    return stats(c)

# ------------------------------------------------------------------------------------------------------ stage f: decision
def pre_decide(st, T0, thr):
    """Disguises and fp8, read before the fingerprint. Returns flags per endpoint."""
    eps = sorted(st); pre = {}
    for e in eps:
        c = st[e]['cal'] or {}; p = st[e]['plain']; r = st[e]['rep']; f8 = (st[e].get('fp8') or {}).get('reset') or {}
        d = dict(T=c.get('T'), z0=c.get('z0'))
        # watermark: identical requests repeat themselves. Many repeated prompts: Fisher z over their p-values. Few (one designed prompt sent
        # 8 times): the Fisher z is capped by the Monte Carlo floor, so one prompt at the floor, far outside the simulated spread, is enough.
        rb = r.get('best') or {}
        d['watermark'] = bool(r['fz'] >= thr['wm_fz'] or ((r.get('beyond') or rb.get('p', 1) <= thr['wm_p']) and r.get('excess', 0) >= thr['wm_excess'])) \
            if r['groups_tested'] else bool(r['max_lcp'] >= thr['wm_lcp'])
        # replaced system context: the plain-context short answers are better explained without the default system prompt (likelihood ratio),
        # or the extraction prompts are far in the tail of the simulated base null at the endpoint's own temperature
        # replaced system context: the plain-context short answers are far less likely than the base predicts at the temperature fitted on the
        # endpoint's own reset-prefixed long answers (persona: z <= plain_z_strong), or moderately so while an alternative system prompt
        # explains the extraction prompts better than the default one (system prompt: z <= plain_z and likelihood ratio >= alt_llr)
        zp = p['z_plain_own'] if p['n_plain'] else 0.0; al = p.get('alt_llr_probe')
        d['prompt'] = bool(zp <= thr['plain_z_strong'] or (zp <= thr['plain_z'] and al is not None and al >= thr['alt_llr']))
        d['prompt_why'] = (f"plain answers z {zp} at the temperature of its reset-prefixed answers, alternative system prompt better by {al} nats on the "
                           f"extraction prompts, extraction prompts z {p['probe_fz_own']} against the simulated base")
        tr = st[e].get('trunc') or {}
        d['trunc'] = bool(tr.get('detected')); d['trunc_why'] = tr.get('worst')
        d['hot'] = bool(c.get('z0') is not None and c['z0'] <= thr['hot_z'])
        d['cold'] = bool(c.get('z0') is not None and c['z0'] >= thr['cold_z'])
        d['fp8_z0'] = f8.get('z0'); d['fp8_frac'] = f8.get('frac')
        d['fp8'] = bool(f8 and f8['z0'] >= thr['fp8_z0'] and f8['frac'] >= thr['fp8_frac'])
        d['fp8_why'] = f"recipe {(st[e].get('fp8') or {}).get('recipe')}: likelihood ratio z {f8.get('z0')} above the base expectation, {f8.get('z1')} from the quantized expectation, fraction {f8.get('frac')} of the way" if f8 else ''
        d['explained'] = d['watermark'] or d['prompt'] or d['trunc'] or d['fp8']
        pre[e] = d
    return pre

def decide(st, pre, gfp, lora, T0, thr):
    """Rules, in order (no endpoint id anywhere):
       1. watermark (repeatability)            -> base model (disguise)
       2. replaced system context              -> base model (disguise; it bends the fingerprint as much as an adapter, so it is read first)
       2b. truncated sampling (top-k / top-p)  -> base model (disguise)
       3. fp8 (calibrated likelihood ratio)    -> trap
       4. fingerprint self z >= gfp_self       -> adapter trap
       5. calibrated p <= gfp_p along one of the two strongest adapter endpoints, slope > 0 -> weaker dose of that adapter, trap
       6. temperature away from the family, fingerprint clean -> base model sampled at another temperature (disguise)
       7. everything else                      -> base model"""
    eps = sorted(st); out = {}
    g = gfp.get(thr['gfp_rows']) if gfp else None
    # a keyed sampler leaves the weights alone, so a watermark endpoint has a clean fingerprint; a sharper adapter can repeat SHORT prefixes
    # (simulated LoRA 0.5: repeatability z 4.1 with a 3-character prefix). Keep the watermark call only with a long shared prefix or a clean fingerprint.
    if g:
        for e in eps:
            if pre[e]['watermark'] and g['self'][e]['z'] > 0 and g['self'][e]['p'] <= 0.01 \
                    and (st[e]['rep'].get('max_lcp') or 0) < thr['wm_lcp']:
                pre[e]['watermark'] = False
                pre[e]['wm_dropped'] = f"repeatability without a long shared prefix ({st[e]['rep'].get('max_lcp')} characters) and fingerprint self p {g['self'][e]['p']}"
                pre[e]['explained'] = pre[e]['prompt'] or pre[e]['trunc'] or pre[e]['fp8']
    disg = {e: bool(pre[e]['watermark'] or pre[e]['prompt'] or pre[e].get('trunc')) for e in eps}
    free = [e for e in eps if not disg[e] and not pre[e]['fp8']]
    strong, weak = [], {}
    if g:
        # an endpoint whose temperature is away from the family needs less: a pure change of temperature leaves the fingerprint at zero
        strong = [e for e in free if g['self'][e]['z'] > 0 and g['self'][e]['p'] <= (thr['gfp_self_p_hot'] if (pre[e]['hot'] or pre[e]['cold']) else thr['gfp_self_p'])]
        def link(a, b):                                    # calibrated: tests a along the direction of b (not symmetric)
            p = g['pair'][a][b]
            return p['p'] is not None and p['p'] <= thr['gfp_p'] and (p['ratio'] or 0) > 0
        # directions = the two strongest doses (largest self effect); without any endpoint above gfp_self, the two largest self z above gfp_ref_min
        refs = sorted(strong, key=lambda e: -g['self'][e]['effect'])[:2] or \
               sorted((e for e in free if g['self'][e]['z'] >= thr['gfp_ref_min']), key=lambda e: -g['self'][e]['z'])[:2]
        for e in free:
            if e in strong: continue
            cands = [r for r in refs if r != e and link(e, r)]
            if cands: weak[e] = min(cands, key=lambda r: g['pair'][e][r]['p'])
        # truncated sampling explains how tokens are drawn, not the weights: a LoRA dose can be served with top_k too. The weaker-dose test
        # still applies to a truncated endpoint (simulation: base + top_k stays at slope +-0.03, p >= 0.06; LoRA + top_k keeps its slope).
        for e in eps:
            if pre[e].get('trunc') and not pre[e]['prompt'] and not pre[e]['watermark'] and not pre[e]['fp8']:
                cands = [r for r in refs if r != e and link(e, r)]
                if cands: weak[e] = min(cands, key=lambda r: g['pair'][e][r]['p']); pre[e]['trunc_lora'] = True
    for e in eps:
        d = pre[e]; c = st[e]['cal'] or {}; why = []; kind = 'base'
        gs = g['self'][e] if g else None
        best = max(((o, g['pair'][e][o]) for o in eps if o != e), key=lambda kv: kv[1]['z']) if g and len(eps) > 1 else None
        if g and len(eps) > 1: best = min(((o, g['pair'][e][o]) for o in eps if o != e), key=lambda kv: kv[1]['p'] if kv[1]['p'] is not None else 1.0)
        fing = f"fingerprint self p {gs['p']} (z {gs['z']}), smallest calibrated p {best[1]['p']} along {best[0]}" if gs and best else 'fingerprint not run'
        if d['watermark']:
            rb = st[e]['rep'].get('best') or {}
            why.append(f"watermark (identical requests share {rb.get('prefix')} characters, at most {rb.get('largest_simulated')} in 2000 simulated draws of the base, excess {rb.get('excess')} sd, Fisher z over prompts {st[e]['rep']['fz']})")
        if d['prompt']: why.append(f"replaced system context ({d.get('prompt_why', '')})")
        if d.get('trunc'): why.append(f"truncated sampling at T {c.get('T')} ({d.get('trunc_why')})")
        if d['fp8']:
            kind = 'fp8'; why.append(f"fp8 ({d.get('fp8_why', '')})")
        elif not disg[e] and e in strong:
            kind = 'lora'; why.append(f"adapter: fingerprint self test p {gs['p']} (z {gs['z']}), T {c.get('T')}")
        elif (not disg[e] or d.get('trunc_lora')) and e in weak:
            p = g['pair'][e][weak[e]]
            kind = 'lora'; why.append(f"adapter, weaker dose{' served with truncated sampling' if d.get('trunc_lora') else ''}: calibrated p {p['p']} along {weak[e]} (t {p['t']}), dose ratio {p['ratio']} (se {p['ratio_se']}); self z {gs['z']}")
        elif not disg[e] and (d['hot'] or d['cold']):
            clean = gs is not None and (gs['z'] <= 0 or gs['p'] > 0.05)
            why.append((f"sampling disguise: T {c.get('T')} against {T0}, " if clean else f"T {c.get('T')} against {T0}, deviation below the trap thresholds, ") + fing)
        if kind == 'base' and not why: why.append(f"no trap signal (T {c.get('T')}, z0 {c.get('z0')}, {fing})")
        elif kind == 'base' and disg[e]: why.append(fing)
        out[e] = dict(kind=kind, why='; '.join(why))
    # composition prior (learned from the practice set: one deployment per configuration): a family has at most `max_plain` endpoints
    # that are the base model with no disguise at all. When more endpoints than that show neither a disguise nor a trap, the most
    # base-like ones stay in the base group and the others become models of their own ("unresolved": a trap no test of ours can see).
    def deviation(e):
        """How far from a plain base endpoint: squares of the stop test, the quantization test (both two-sided), the fingerprint self test
        and the strongest pull along an adapter direction (both one-sided). Each is about N(0,1) for the plain base endpoint."""
        c = st[e]['cal'] or {}; f8 = (st[e].get('fp8') or {}).get('reset') or {}
        parts = {'stop': c.get('z_stop') or 0.0, 'quantization': f8.get('z0') or 0.0}
        if g:
            parts['fingerprint self'] = max(0.0, g['self'][e]['z'])
            al = [(g['pair'][e][o]['t'] or 0.0, o) for o in eps if o != e and (o in strong or o in weak)]
            if al: t, o = max(al); parts['along ' + o] = max(0.0, t)
        k = max(parts, key=lambda x: abs(parts[x]))
        return round(sum(v * v for v in parts.values()), 2), f'{k} {parts[k]:+.1f}', {x: round(v, 2) for x, v in parts.items()}
    plain = [e for e in eps if out[e]['kind'] == 'base' and not disg[e] and not (pre[e]['hot'] or pre[e]['cold'])]
    dev = {e: deviation(e) for e in plain}
    for e in plain: out[e]['plain_candidate'] = dict(sum_squares=dev[e][0], largest=dev[e][1], parts=dev[e][2])
    ranked = sorted(plain, key=lambda e: dev[e][0])
    for i, e in enumerate(ranked):
        out[e]['why'] += f"; plain candidate {i + 1} of {len(ranked)}: sum of squares of its base-model statistics {dev[e][0]} ({len(dev[e][2])} expected), largest {dev[e][1]}"
        if i >= int(thr['max_plain']):
            out[e]['kind'] = 'unres'
            out[e]['why'] += f"; not kept in the base group: at most {int(thr['max_plain'])} endpoint without any disguise per family, {ranked[0]} is more base-like ({dev[ranked[0]][0]})"
    # two adapter traps are the same model when they deviate in the same direction with a dose ratio of about 1 at the same temperature
    traps = [e for e in eps if out[e]['kind'] != 'base']
    label = {e: e for e in traps}
    for a, b in itertools.combinations([e for e in traps if out[e]['kind'] == 'lora'], 2):
        p, q = g['pair'][a][b], g['pair'][b][a]
        Ta, Tb = (st[a]['cal'] or {}).get('T'), (st[b]['cal'] or {}).get('T')
        if p['p'] is not None and q['p'] is not None and max(p['p'], q['p']) <= thr['gfp_p'] and p['cos'] is not None and p['cos'] >= thr['same_cos'] and p['ratio'] and q['ratio'] \
           and all(thr['same_lo'] <= x <= thr['same_hi'] for x in (p['ratio'], q['ratio'])) and Ta is not None and Tb is not None and abs(Ta - Tb) <= thr['same_dT']:
            tgt = label[a]
            for k in label:
                if label[k] == label[b]: label[k] = tgt
            out[b]['why'] += f"; same model as {a} (cosine {p['cos']}, dose ratios {p['ratio']} and {q['ratio']}, T {Ta} and {Tb})"
    for e in eps: out[e]['model'] = 'base' if out[e]['kind'] == 'base' else 'trap:' + label[e]
    return out

def p_base(e, st, pre, gfp, dec, thr):
    """rough probability that the endpoint serves the plain base weights (used for the pair confidence only)"""
    sig = lambda x: 1 / (1 + math.exp(-max(-30, min(30, x))))
    p_trap = 0.03
    if pre[e].get('fp8_z0') is not None: p_trap = max(p_trap, sig(pre[e]['fp8_z0'] - thr['fp8_z0']))
    g = gfp.get(thr['gfp_rows']) if gfp else None
    if g and not (pre[e]['watermark'] or pre[e]['prompt'] or pre[e].get('trunc')):
        if g['self'][e]['z'] > 0: p_trap = max(p_trap, sig(math.log(thr['gfp_self_p'] / max(g['self'][e]['p'], 1e-6))))
        for o, p in g['pair'][e].items():
            if dec.get(o, {}).get('kind') == 'lora' and p['p'] is not None and (p['ratio'] or 0) > 0:
                p_trap = max(p_trap, sig(math.log(thr['gfp_p'] / max(p['p'], 1e-6))))
    if dec.get(e, {}).get('kind') == 'unres': p_trap = max(p_trap, 0.6)          # kept apart by the composition prior only
    elif dec.get(e, {}).get('kind') == 'base' and 'plain_candidate' in dec.get(e, {}) and \
            sum(1 for v in dec.values() if 'plain_candidate' in v) > thr['max_plain']: p_trap = max(p_trap, 0.35)   # best of several plain candidates
    return min(0.99, max(0.01, 1 - p_trap))

# --------------------------------------------------------------------------------------------------------------- output
def mcc(pred, truth):
    tp = tn = fp = fn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        if a not in pred or b not in pred: continue
        t = truth[a] == truth[b]; p = pred[a] == pred[b]
        tp += t and p; tn += (not t) and (not p); fp += (not t) and p; fn += t and (not p)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / den if den else 0.0), dict(tp=tp, tn=tn, fp=fp, fn=fn)

def score_truth(part, truth_csv):
    tf, tm = {}, {}
    for r in csv.DictReader(open(truth_csv)):
        tf[r['endpoint']] = r['family']; tm[r['endpoint']] = (r['family'], r['model'])
    pf = {e: v['family'] for e, v in part.items()}; pm = {e: (v['family'], v['model']) for e, v in part.items()}
    mf, cf = mcc(pf, tf)
    # model MCC only over pairs of the same true family
    tp = tn = fp = fn = 0
    for a, b in itertools.combinations(sorted(tm), 2):
        if a not in pm or b not in pm or tf[a] != tf[b]: continue
        t = tm[a] == tm[b]; p = pm[a] == pm[b]
        tp += t and p; tn += (not t) and (not p); fp += (not t) and p; fn += t and (not p)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); mm = (tp * tn - fp * fn) / den if den else 0.0
    return dict(mcc_family=round(max(0, mf), 4), mcc_model=round(max(0, mm), 4), score=round(0.1 * max(0, mf) + 0.9 * max(0, mm), 4),
                model_pairs=dict(tp=tp, tn=tn, fp=fp, fn=fn))

def write_outputs(outdir, fam, fams, results, cfg, truth=None):
    thr = cfg['thr']
    part = {}; diag = dict(config={k: v for k, v in cfg.items() if k != 'thr'}, thresholds=thr, families={})
    md = ['# Model Equality: diagnosis', '']
    pairs = []
    for fi, (repo, R) in enumerate(sorted(results.items())):
        fl = f'F{fi + 1}'
        st, pre, gfp, lora, dec, T0 = R['st'], R['pre'], R['gfp'], R['lora'], R['dec'], R['T0']
        g = gfp.get(thr['gfp_rows']) if gfp else None
        eps = sorted(st)
        def best_pair(e, which):
            gg = gfp.get(which) if gfp else None
            if not gg or len(eps) < 2: return None
            o, p = min(gg['pair'][e].items(), key=lambda kv: kv[1]['p'] if kv[1]['p'] is not None else 1.0); return dict(partner=o, **p)
        diag['families'][fl] = dict(base=repo, template_date=R['date'], T0=T0, endpoints={e: dict(
            family=fam[e], measures=st[e], flags=pre[e], decision=dec[e],
            fingerprint={w: dict(self=gfp[w]['self'][e], pairs=gfp[w]['pair'][e]) for w in gfp} if gfp else None,
            adapter_self=(lora or {}).get('self', {}).get(e), adapter_pooled=(lora or {}).get('pooled', {}).get(e)) for e in eps},
            adapter=dict(train=(lora or {}).get('train'), confirmed=(lora or {}).get('confirmed'), cross=(lora or {}).get('cross')))
        md += [f'## Family {fl}: base {repo}, {len(eps)} endpoints, family temperature {T0}' + (f', template date {R["date"]}' if R['date'] else ''), '']
        base = [e for e in eps if dec[e]['kind'] == 'base']
        md += [f'Base model group ({len(base)}): {" ".join(base)}', '']
        groups = collections.defaultdict(list)
        for e in eps:
            if dec[e]['kind'] != 'base': groups[dec[e]['model']].append(e)
        md += ['Traps: ' + (', '.join('{' + ' '.join(v) + '} ' + dec[v[0]]['kind'] for v in groups.values()) or 'none'), '']
        md += ['| endpoint | answers long/plain/repeat | T | z at T0 | stop z | truncation z | plain answers z / alt system LLR on extraction | repeat excess sd (prefix) | fp8 z0 / z1 / frac | fingerprint self z, sign-flip p | smallest calibrated p: along, p, dose | decision | why |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for e in eps:
            c = st[e]['cal'] or {}; p = st[e]['plain']; r = st[e]['rep']; f8 = (st[e].get('fp8') or {}).get('reset') or {}
            bp = best_pair(e, thr['gfp_rows'])
            md.append(f"| {e} | {c.get('n_ans', 0)}/{p['n_plain']}/{r['n_repeat_answers']} | {c.get('T')} | {c.get('z0')} | {c.get('z_stop')} | {(st[e].get('trunc') or {}).get('z')} | {p['z_plain_own']} / {p.get('alt_llr_probe')} | "
                      f"{r.get('excess')} ({r['max_lcp']}) | {f8.get('z0')} / {f8.get('z1')} / {f8.get('frac')} | "
                      + (f"{g['self'][e]['z']}, {g['self'][e]['p']}" if g else '-') + ' | '
                      + (f"{bp['partner']}: {bp['p']}, {bp['ratio']}" if bp else '-') + f" | {dec[e]['kind']} | {dec[e]['why']} |")
        md.append('')
        if g:
            md += ['Calibrated p x 1000 (row = endpoint under test, column = direction; the other way round is another test), answers = ' + thr['gfp_rows'] + ':', '',
                   '| | ' + ' | '.join(e[-2:] for e in eps) + ' |', '|---' * (len(eps) + 1) + '|']
            md += [f"| {a[-2:]} | " + ' | '.join('.' if a == b else (f"{1000 * g['pair'][a][b]['p']:.0f}" if g['pair'][a][b]['p'] is not None else '-') for b in eps) + ' |' for a in eps] + ['']
        if lora and lora.get('pooled'):
            md += [f"Adapter proxy (corroboration only): trained on {' '.join(lora.get('train') or []) or 'nothing'}; cross-fitted gain per answer (z): " +
                   ', '.join(f"{e} {v['cf_gain_per_answer']} ({v['cf_z']})" for e, v in sorted(lora['pooled'].items())), '']
        pb = {e: p_base(e, st, pre, gfp, dec, thr) for e in eps}
        for e in eps: part[e] = dict(family=fl, model=fl + ('-base' if dec[e]['kind'] == 'base' else '-' + dec[e]['model'].replace('trap:', 'trap-')))
        for a, b in itertools.combinations(eps, 2):
            same = part[a]['model'] == part[b]['model']
            q = 0.9 if (same and dec[a]['kind'] != 'base') else 0.05
            ps = pb[a] * pb[b] + (1 - pb[a]) * (1 - pb[b]) * q
            pairs.append((a, b, int(same), round(ps if same else 1 - ps, 3)))
    with open(os.path.join(outdir, 'partition.csv'), 'w') as f:
        f.write('endpoint,family,model\n')
        for e in sorted(part): f.write(f"{e},{part[e]['family']},{part[e]['model']}\n")
    bym = collections.defaultdict(list)
    for e in sorted(part): bym[part[e]['model']].append(e)
    open(os.path.join(outdir, 'groups.txt'), 'w').write(';'.join('+'.join(v) for v in bym.values() if len(v) > 1) + '\n')   # argument of hk/mine/submit.py
    with open(os.path.join(outdir, 'pairs.csv'), 'w') as f:
        f.write('endpoint_a,endpoint_b,same_model,confidence\n')
        for p in pairs: f.write(','.join(str(x) for x in p) + '\n')
    if truth:
        sc = score_truth(part, truth); diag['score'] = sc
        md += [f"## Score against {os.path.basename(truth)}", '', f"family MCC {sc['mcc_family']}, model MCC {sc['mcc_model']}, score {sc['score']}, model pairs {sc['model_pairs']}", '']
        log('SCORE', json.dumps(sc))
    diag['partition'] = part
    json.dump(diag, open(os.path.join(outdir, 'diagnosis.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    open(os.path.join(outdir, 'diagnosis.md'), 'w').write('\n'.join(md) + '\n')
    return part, diag

# ----------------------------------------------------------------------------------------------------------------- main
def load_config(path, overrides=None):
    cfg = json.load(open(path)) if path else {}
    cfg['candidates'] = [c['repo'] if isinstance(c, dict) else c for c in cfg.get('candidates', [])]
    cfg.setdefault('reset_markers', ['<|im_start|>system', '<|start_header_id|>system'])
    cfg.setdefault('probe_prefixes', ['per-'])
    thr = dict(THR); thr.update(cfg.get('thr', {})); thr.update(overrides or {}); cfg['thr'] = thr
    return cfg

def run(cfg, rows, outdir, cache_dir=None, truth=None, no_lora=False, no_cross=False, redecide=False, proxy=False):
    os.makedirs(outdir, exist_ok=True)
    cache = Cache(cache_dir or os.path.join(outdir, 'cache'))
    rows = annotate(rows, cfg)
    log(f'{len(rows)} answers, {len({r["endpoint"] for r in rows})} endpoints')
    fam, dates = stage_family(rows, cfg, cache)
    fams = collections.defaultdict(list)
    for e, v in fam.items(): fams[v['base']].append(e)
    results = {}
    state_p = os.path.join(outdir, 'state.pkl')
    state = pickle.load(open(state_p, 'rb')) if redecide and os.path.exists(state_p) else {}
    for repo, eps in sorted(fams.items()):
        log(f'family {repo}: {len(eps)} endpoints: {" ".join(sorted(eps))}')
        frows = [r for r in rows if r['endpoint'] in set(eps)]
        if repo in state:
            S = state[repo]; st, T0, gfp, lora = S['st'], S['T0'], S.get('gfp'), S.get('lora'); dates[repo] = S.get('date')
        else:
            st, T0, dates[repo] = measure_family(repo, dates.get(repo), frows, cfg, cache); gfp = lora = None
        pre = pre_decide(st, T0, cfg['thr'])
        if repo not in state and not no_lora:
            gfp = stage_gradfp(repo, dates.get(repo), frows, st, T0, cfg, cache)
            if proxy: lora = stage_lora(repo, dates.get(repo), frows, st, pre, cfg, do_cross=not no_cross)
        dec = decide(st, pre, gfp, lora, T0, cfg['thr'])
        results[repo] = dict(st=st, pre=pre, gfp=gfp, lora=lora, dec=dec, T0=T0, date=dates.get(repo))
        state[repo] = dict(st=st, T0=T0, gfp=gfp, lora=lora, date=dates.get(repo))
        pickle.dump(state, open(state_p, 'wb'))
        for e in sorted(eps): log(f'   {e}: {dec[e]["kind"]:5s} {dec[e]["why"]}')
    return write_outputs(outdir, fam, fams, results, cfg, truth)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True); ap.add_argument('--answers', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--cache', default=None, help='directory of per-answer measurements, reused across runs (default OUT/cache)')
    ap.add_argument('--truth', default=None, help='optional CSV endpoint,family,model: prints the MCC')
    ap.add_argument('--no-lora', action='store_true', help='skip stage e (gradient fingerprint, and the adapter proxy if asked)')
    ap.add_argument('--proxy', action='store_true', help='also run the adapter proxy (slower; reported next to the fingerprint, not used by the decision)')
    ap.add_argument('--no-cross', action='store_true', help='with --proxy: skip the cross test between adapter traps')
    ap.add_argument('--redecide', action='store_true', help='reuse OUT/state.pkl (measurements and adapter stage) and only redo the decision')
    ap.add_argument('--set', nargs='*', default=[], help='threshold overrides name=value')
    a = ap.parse_args()
    ov = {}
    for s in a.set:
        k, v = s.split('='); ov[k] = v if isinstance(THR.get(k), str) else type(THR[k])(float(v)) if k in THR else float(v)
    cfg = load_config(a.config, ov)
    rows = load_rows(a.answers, set(cfg['endpoints']) if cfg.get('endpoints') else None)
    run(cfg, rows, a.out, a.cache, a.truth, a.no_lora, a.no_cross, a.redecide, a.proxy)

if __name__ == '__main__':
    main()
