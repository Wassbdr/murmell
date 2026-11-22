"""Known-key green-list watermark sweep on the scored endpoints (and refs if files given). z-score per endpoint for several public schemes."""
import json, sys, collections, math, torch
from transformers import AutoTokenizer, AutoConfig
MOD = {'QA': 'Qwen/Qwen2.5-3B-Instruct', 'QB': 'Qwen/Qwen2.5-1.5B-Instruct', 'LL': 'unsloth/Llama-3.2-1B-Instruct', 'RF': 'HuggingFaceTB/SmolLM2-1.7B-Instruct'}
F = {'QA': '01,02,04,12,14,17,18,23,24,25,33', 'QB': '03,06,07,08,20,21,22,26,28,30,32', 'LL': '05,09,10,11,13,15,16,19,27,29,31'}
fam = {'ep-' + n: f for f, s in F.items() for n in s.split(',')}
for i in range(1, 10): fam[f'ref-0{i}'] = 'RF'
KEY = 15485863; DEV = sys.argv.pop(1)
rows = []
for p in sys.argv[1:]:
    for l in open(p):
        r = json.loads(l)
        if r.get('endpoint') in fam and r.get('response'): rows.append((r['endpoint'], r['response']))
print('rows', len(rows), flush=True)
for f in ('RF', 'QB', 'QA', 'LL'):
    eps = sorted({e for e, _ in rows if fam[e] == f})
    if not eps: continue
    tok = AutoTokenizer.from_pretrained(MOD[f]); cfg = AutoConfig.from_pretrained(MOD[f]); Vs = sorted({cfg.vocab_size, len(tok)})
    pairs = collections.defaultdict(list)   # endpoint -> list of (prev, cur)
    for e, txt in rows:
        if fam[e] != f: continue
        ids = tok(txt, add_special_tokens=False).input_ids
        pairs[e] += list(zip(ids[:-1], ids[1:]))
    allp = collections.defaultdict(set)
    for e in pairs:
        for a, b in pairs[e]: allp[a].add(b)
    print(f, 'endpoints', len(eps), 'tokens/endpoint', {e[-2:]: len(pairs[e]) for e in eps}, 'distinct prev', len(allp), 'vocab sizes', Vs, flush=True)
    for dev in (DEV,):
        for V in Vs:
            for scheme in ('lefthash', 'markllm'):
                rng = torch.Generator(device=dev); pos = {}
                if scheme == 'markllm':
                    rng.manual_seed(KEY); prf = torch.randperm(V, device=dev, generator=rng)
                for a, bs in allp.items():
                    if scheme == 'lefthash': seed = (KEY * a) % (2 ** 64 - 1)
                    else: seed = (KEY * int(prf[a % V].item())) % V
                    rng.manual_seed(seed); perm = torch.randperm(V, device=dev, generator=rng)
                    inv = torch.empty(V, dtype=torch.long, device=dev); inv[perm] = torch.arange(V, device=dev)
                    bl = [b for b in bs if b < V]
                    for b, p_ in zip(bl, inv[torch.tensor(bl, device=dev)].tolist()): pos[(a, b)] = p_ / V
                for g in (0.25, 0.5):
                    zs = {}
                    for e in eps:
                        ps = [pos[x] for x in set(pairs[e]) if x in pos]; T = len(ps); G = sum(p_ < g for p_ in ps)
                        zs[e] = (G - g * T) / math.sqrt(T * g * (1 - g)) if T else 0.0
                    top = sorted(zs.items(), key=lambda kv: -kv[1])
                    flag = 'HIT ' if top[0][1] > 5 else ''
                    print(f"{flag}{f} dev={dev} V={V} {scheme} gamma={g}: " + ' '.join(f"{e[-2:]}:{z:+.1f}" for e, z in top), flush=True)
