"""Measurements of the task-2 pipeline on toy endpoints, in the structures read by pipeline/me_pipeline.py.

Each function computes, with exact toy likelihoods, the statistic of one stage of the real pipeline; the decision itself
is the pipeline's own code: me_pipeline.pre_decide (disguises, fp8) then me_pipeline.decide (fingerprint, doses,
composition prior), and the score is me_pipeline.score_truth (MCC over pairs, 10 % family + 90 % model).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import me_pipeline as mp  # noqa: E402

from .toy import T0, V, q8, softmax  # noqa: E402

TG = np.array(mp.TG)


def first_prev(prompt):
    return int(hashlib.md5(prompt.encode()).hexdigest(), 16) % V


def pairs(rows, kinds):
    """(prev, next, row index) for every generated token of the rows of the given kinds."""
    P, X, R = [], [], []
    for j, r in enumerate(rows):
        if r["kind"] not in kinds:
            continue
        prev = first_prev(r["prompt"])
        for x in r["toks"]:
            P.append(prev); X.append(x); R.append(j); prev = x
    return np.array(P), np.array(X), np.array(R)


def logp_table(W, b, T):
    z = (W + b[None]) / T
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def z_at(W, b, T, P, X):
    """likelihood z of the tokens against their expectation under the model at T (about N(0,1) when it wrote them)."""
    L = logp_table(W, b, T); p = np.exp(L)
    E = (p * L).sum(1); Var = (p * L ** 2).sum(1) - E ** 2
    return float((L[P, X] - E[P]).sum() / np.sqrt(Var[P].sum()))


def family_ll(base, P, X):
    """stage a: best mean log-likelihood per token over the temperature grid."""
    return max(float(logp_table(base.W, base.b_default, T)[P, X].mean()) for T in TG)


def calibrate(base, P, X):
    S = np.array([logp_table(base.W, base.b_default, T)[P, X].sum() for T in TG]); i = int(np.argmax(S)); T = TG[i]
    if 0 < i < len(TG) - 1:
        y0, y1, y2 = S[i - 1:i + 2]; den = y0 - 2 * y1 + y2
        if den < 0:
            T = TG[i] + 0.5 * (y0 - y2) / den * (TG[1] - TG[0])
    return dict(T=round(float(T), 3), z0=round(z_at(base.W, base.b_default, T0, P, X), 2), z_stop=0.0)


def truncation(base, T, P, X, ks=mp.KS):
    L = logp_table(base.W, base.b_default, T); p = np.exp(L)
    rank = (L[P] > L[P, X][:, None]).sum(1)                     # 0 = most likely token
    srt = np.sort(p, 1)[:, ::-1]; worst = None; det = False
    for k in ks:
        exp = float(srt[P, k:].sum()); obs = int((rank >= k).sum())
        if exp >= mp.THR["trunc_min_exp"]:
            r = obs / exp
            if worst is None or r < worst["ratio"]:
                worst = dict(k=k, observed=obs, expected=round(exp, 1), ratio=round(r, 3))
            det |= r <= mp.THR["trunc_ratio"]
    return dict(detected=bool(det), worst=worst)


def sample_fast(base, T, prompt, n_tok, n, rng):
    p = softmax((base.W + base.b_default[None]) / T); c = np.cumsum(p, 1)
    out = np.zeros((n, n_tok), int); prev = np.full(n, first_prev(prompt))
    for t in range(n_tok):
        u = rng.random(n)[:, None]; x = np.minimum((c[prev] < u).sum(1), V - 1); out[:, t] = x; prev = x
    return [list(r) for r in out]


def repeatability(base, T, rows, seed):
    rng = np.random.default_rng(seed); ps = []; best = None; mx = 0
    groups = {}
    for r in rows:
        if r["kind"] == "repeat":
            groups.setdefault(r["prompt"], []).append(r["toks"])
    for pr, ans in sorted(groups.items()):
        null = sample_fast(base, T, pr, len(ans[0]), mp.THR["null_n"], rng)
        obs, p, exc, big = mp.rep_pvalue(ans, null, R=2000, seed=seed)
        ps.append(p); mx = max(mx, obs)
        if best is None or p < best["p"]:
            best = dict(prompt=pr, prefix=obs, p=round(p, 5), excess=round(exc, 2), largest_simulated=big)
    return dict(groups=len(groups), groups_tested=len(ps), max_lcp=mx, fz=round(mp.fisher_z(ps), 2), best=best,
                beyond=bool(best and best["prefix"] > best["largest_simulated"]), excess=best["excess"] if best else 0)


def fp8_test(base, T, P, X):
    """likelihood ratio between our own fp8 copy of the base and the base, with its mean and sd under both hypotheses."""
    Lb = logp_table(base.W, base.b_default, T); Lq = logp_table(q8(base.W), base.b_default, T)
    D = Lq - Lb; pb, pq = np.exp(Lb), np.exp(Lq)
    llr = float(D[P, X].sum())
    E0 = (pb * D).sum(1)[P].sum(); E1 = (pq * D).sum(1)[P].sum()
    V0 = ((pb * D ** 2).sum(1) - (pb * D).sum(1) ** 2)[P].sum(); V1 = ((pq * D ** 2).sum(1) - (pq * D).sum(1) ** 2)[P].sum()
    return dict(z0=round(float((llr - E0) / np.sqrt(V0)), 2), z1=round(float((llr - E1) / np.sqrt(V1)), 2),
                frac=round(float((llr - E0) / (E1 - E0)), 3))


def sketch(seed, R=64):
    r = np.random.default_rng(seed)
    return r.normal(0, 1, (R, V)), r.normal(0, 1, (R, V))


def grad_folds(base, T, rows, P, X, Rw, U, Wv, K):
    """gradient fingerprint: score of each token's log-likelihood w.r.t. W, read on a random rank-1 sketch
    (u_r v_r^T); summed per prompt fold (folds of the real pipeline: hash of the prompt)."""
    p = softmax((base.W + base.b_default[None]) / T)
    g_tok = U[:, P] * (Wv[:, X] - (p[P] @ Wv.T).T) / T            # R x n_tokens
    fold = np.array([int(hashlib.md5(rows[j]["prompt"].encode()).hexdigest(), 16) % K for j in Rw])
    Gf = np.zeros((K, U.shape[0])); nt = np.zeros(K)
    for k in range(K):
        Gf[k] = g_tok[:, fold == k].sum(1); nt[k] = (fold == k).sum()
    return Gf, nt


def measure(eps, answers, bases, seed=0):
    """answers: name -> rows. Returns family per endpoint, the family log-likelihood table and st per family."""
    fams = sorted(bases)
    ll = {e.name: {f: family_ll(bases[f], *pairs(answers[e.name], ("long",))[:2]) for f in fams} for e in eps}
    fam_of = {n: max(v, key=v.get) for n, v in ll.items()}
    st = {f: {} for f in fams}
    for e in eps:
        rows = answers[e.name]; b = bases[fam_of[e.name]]
        P, X, _ = pairs(rows, ("long",)); Pp, Xp, _ = pairs(rows, ("plain",))
        cal = calibrate(b, P, X)
        st[fam_of[e.name]][e.name] = dict(
            cal=cal,
            plain=dict(n_plain=int(len(Pp) > 0), z_plain_own=round(z_at(b.W, b.b_default, cal["T"], Pp, Xp), 2),
                       alt_llr_probe=None, probe_fz_own=None),
            rep=repeatability(b, cal["T"], rows, seed), trunc=truncation(b, cal["T"], P, X),
            fp8=dict(reset=fp8_test(b, cal["T"], P, X), recipe="toy_q8"))
    return fam_of, ll, st


def fingerprint(fam_eps, answers, base, st, K=None, seed=0):
    K = K or mp.THR["gfp_folds"]; U, Wv = sketch(seed)
    blocks, nts = [], []
    for n in fam_eps:
        rows = answers[n]; P, X, Rw = pairs(rows, ("long", "repeat"))
        Gf, nt = grad_folds(base, st[n]["cal"]["T"], rows, P, X, Rw, U, Wv, K); blocks.append(Gf); nts.append(nt)
    A = np.concatenate(blocks); G = A @ A.T
    return mp.gfp_stats(list(fam_eps), K, G, np.concatenate(nts))


def decide_family(st_f, gfp):
    pre = mp.pre_decide(st_f, T0, mp.THR)
    return mp.decide(st_f, pre, {"all": gfp}, None, T0, mp.THR), pre
