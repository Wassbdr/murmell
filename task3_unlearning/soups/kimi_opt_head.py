"""Calibre la tete d'un modele avec le simulateur local de Lyna.

Optimise une echelle globale et un biais par classe sur les cinq premieres
configurations, puis mesure la generalisation sur les configurations restantes.
Aucun modele n'est sauvegarde si la validation ne s'ameliore pas.

Depuis /workspace/unlearning :
  python tools/kimi_opt_head.py --target kimi_soupe_rep --output kimi_soupe_bias
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="kimi_soupe_rep")
    ap.add_argument("--output", default="kimi_soupe_bias")
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--train", type=int, default=5)
    ap.add_argument("--regs", default="1.0,0.3,0.1,0.03,0.01")
    ap.add_argument("--fit", default="_fit.pt")
    ap.add_argument("--virtual_pg", type=float, default=-1.0, help="melange ideal de cible pseudo-gold, sans produire de modele")
    ap.add_argument("--virtual_cond", type=float, default=-1.0, help="correction des classes fausses a perte constante")
    ap.add_argument("--virtual_only", action="store_true")
    ap.add_argument("--save_pg")
    ap.add_argument("--matrix", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = Path("cache/lina_feat")
    meta = torch.load(D / "_meta.pt", weights_only=False)
    fo = torch.load(D / "_orig.pt", weights_only=False)
    raw = torch.load(D / f"{args.target}.pt", weights_only=False)
    top = torch.load(D / args.fit, weights_only=False)
    y = meta["y"].to(dev)
    n = len(y)
    oh = F.one_hot(y, 100).bool()
    fams, kinds = meta["fams"], meta["kinds"]
    groups = ["clean"] + fams
    needed_groups = sorted({"clean", "aug60", "aug40"})

    def tensor(x):
        return x.float().to(dev)

    nm = tensor(fo["nonmem"])
    geo = [0, 1, 2, 3, 10]
    bases = {
        "za": tensor(meta["za"]),
        "nmg": nm[:, geo].mean(1),
        "clean": tensor(fo["clean"]),
        "p2za": 2 * torch.log(tensor(meta["p2"]).clamp_min(1e-9)),
    }
    ce_nm = torch.stack([F.cross_entropy(nm[:, v], y, reduction="none") for v in range(nm.shape[1])], 1)
    diffs = {"aug": tensor(meta["diff"]), "nmg": ce_nm[:, geo].mean(1), "nmall": ce_nm.mean(1)}

    def rank01(d):
        r = torch.empty(n, device=dev)
        r[d.argsort()] = (torch.arange(n, device=dev, dtype=torch.float32) + 0.5) / n
        return r

    ranks = {k: rank01(v) for k, v in diffs.items()}

    def unpack(t):
        sp = F.softplus
        return {
            "mu_c": sp(t[0]), "sd_c": sp(t[1]), "mu_a": sp(t[2]), "sd_a": sp(t[3]),
            "tau": torch.exp(t[4]), "pi_lo": torch.sigmoid(t[5]), "pi_hi": torch.sigmoid(t[6]),
            "qa": torch.sigmoid(t[7]), "beta": sp(t[8]), "betac": sp(t[9]), "gamma": sp(t[10]),
        }

    def prepare(record):
        fam, base, diff = record["cfg"]
        p = unpack(record["t"].to(dev))
        gi = [0, groups.index(fam)]
        lo = torch.stack([tensor(fo[g]) for g in [groups[i] for i in gi]])
        ce_o = torch.stack([F.cross_entropy(z, y, reduction="none") for z in lo])
        lp_o = torch.stack([F.log_softmax(z / 2, 1) for z in lo])
        mu = torch.stack([p["mu_c"], p["mu_a"]])
        sd = torch.stack([p["sd_c"], p["sd_a"]])
        d_o = (ce_o.mean(1) - mu).abs() + (ce_o.std(1) - sd).abs()
        pi = p["pi_lo"] + (p["pi_hi"] - p["pi_lo"]) * torch.sigmoid((p["qa"] - ranks[diff]) / 0.04)
        a = F.softmax((bases[base] / (2 * p["tau"])).masked_fill(oh, -1e4), 1)
        pg = pi[:, None] * oh.float() + (1 - pi[:, None]) * a
        entropy = -(pg * torch.log(pg.clamp_min(1e-12))).sum(1)
        kl_o = -entropy.mean() - torch.einsum("nc,gnc->g", pg, lp_o) / n
        return {"record": record, "gi": gi, "p": p, "mu": mu, "sd": sd, "d_o": d_o.clamp_min(1e-6),
                "pg": pg, "entropy": entropy, "kl_o": kl_o.clamp_min(1e-6)}

    configs = [prepare(r) for r in top]
    zraw = {g: tensor(raw[g]) for g in needed_groups}
    if args.virtual_pg >= 0 or args.virtual_cond >= 0:
        pg = torch.stack([c["pg"] for c in configs[:args.train]]).mean(0)
        if args.save_pg:
            torch.save({"pg": pg.cpu(), "y": y.cpu()}, args.save_pg)
        if args.virtual_pg >= 0:
            for g in zraw:
                q = (1 - args.virtual_pg) * F.softmax(zraw[g] / 2, 1) + args.virtual_pg * pg
                zraw[g] = 2 * torch.log(q.clamp_min(1e-12))
        else:
            pg_alt = pg.masked_fill(oh, 0)
            pg_alt = pg_alt / pg_alt.sum(1, keepdim=True).clamp_min(1e-12)
            for g in zraw:
                p = F.softmax(zraw[g], 1)
                py = p.gather(1, y[:, None])
                p_alt = p.masked_fill(oh, 0)
                p_alt = p_alt / p_alt.sum(1, keepdim=True).clamp_min(1e-12)
                alt = (1 - args.virtual_cond) * p_alt + args.virtual_cond * pg_alt
                q = alt * (1 - py)
                q.scatter_add_(1, y[:, None], py)
                zraw[g] = torch.log(q.clamp_min(1e-12))
    araw = {k: tensor(raw["anc_" + k]) for k in kinds}
    aorig = {k: tensor(fo["anc_" + k]) for k in kinds}
    dp = torch.zeros(100, 100, device=dev)

    def transform(z, scale, bias):
        if args.matrix:
            return z @ (torch.eye(100, device=dev) + dp) + bias
        return scale * z + bias

    def anchor_stats(scale, bias):
        kls, ags, klcs = [], [], []
        for k in kinds:
            zo = aorig[k]
            zs = transform(araw[k], scale, bias)
            lo = F.log_softmax(zo / 2, 1)
            ls = F.log_softmax(zs / 2, 1)
            kl = (lo.exp() * (lo - ls)).sum(1)
            conf = lo.exp().max(1).values > 0.9
            kls.append(kl.mean())
            ags.append((zo.argmax(1) == zs.argmax(1)).float().mean())
            klcs.append(kl[conf].mean() if conf.any() else kl.mean())
        return torch.stack(kls).mean(), torch.stack(ags).mean(), torch.stack(klcs).mean()

    def score(cfg, scale, bias, anchors=None):
        p = cfg["p"]
        mgs, pgs = [], []
        for j, gi in enumerate(cfg["gi"]):
            z = transform(zraw[groups[gi]], scale, bias)
            ce = F.cross_entropy(z, y, reduction="none")
            d_s = (ce.mean() - cfg["mu"][j]).abs() + (ce.std() - cfg["sd"][j]).abs()
            mgs.append(1 - (d_s / cfg["d_o"][j]).clamp(0, 1))
            lp = F.log_softmax(z / 2, 1)
            kl_s = -cfg["entropy"].mean() - torch.einsum("nc,nc->", cfg["pg"], lp) / n
            pgs.append(1 - (kl_s / cfg["kl_o"][j]).clamp(0, 1))
        kla, ag, klc = anchors if anchors is not None else anchor_stats(scale, bias)
        retention = ag.clamp_max(1) ** p["gamma"] / (1 + p["beta"] * kla + p["betac"] * klc)
        return torch.stack(mgs).mean() * torch.stack(pgs).mean() * retention

    def evaluate(scale, bias):
        anchors = anchor_stats(scale, bias)
        values = torch.stack([score(c, scale, bias, anchors) for c in configs])
        return values

    one = torch.tensor(1.0, device=dev)
    zero = torch.zeros(100, device=dev)
    with torch.no_grad():
        baseline = evaluate(one, zero)
    print("baseline", "train", float(baseline[:args.train].mean()), "val", float(baseline[args.train:].mean()),
          "all", float(baseline.mean()), flush=True)
    if args.virtual_only:
        return

    results = []
    for reg in [float(x) for x in args.regs.split(",")]:
        sp = torch.zeros(1, device=dev, requires_grad=True)
        bp = torch.zeros(100, device=dev, requires_grad=True)
        dp = torch.zeros(100, 100, device=dev, requires_grad=True)
        opt = torch.optim.Adam([bp, dp] if args.matrix else [sp, bp], lr=0.025)
        for _ in range(args.steps):
            scale = 1 + 0.4 * torch.tanh(sp)
            bias = 0.5 * torch.tanh(bp)
            anchors = anchor_stats(scale, bias)
            values = torch.stack([score(c, scale, bias, anchors) for c in configs[:args.train]])
            loss = -values.mean() + reg * bias.square().mean() + 0.02 * (scale - 1).square()
            if args.matrix:
                loss = loss + reg * dp.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            scale = 1 + 0.4 * torch.tanh(sp)
            bias = 0.5 * torch.tanh(bp)
            values = evaluate(scale, bias)
            row = {
                "reg": reg, "train": float(values[:args.train].mean()), "val": float(values[args.train:].mean()),
                "all": float(values.mean()), "scale": float(scale), "bias_rms": float(bias.square().mean().sqrt()),
                "bias_max": float(bias.abs().max()), "values": [float(v) for v in values],
                "scale_tensor": scale.detach().cpu(), "bias_tensor": bias.detach().cpu(),
                "matrix_tensor": dp.detach().cpu() if args.matrix else None,
            }
            results.append(row)
            print({k: v for k, v in row.items() if not k.endswith("_tensor") and k != "values"}, flush=True)

    baseline_val = float(baseline[args.train:].mean())
    eligible = [r for r in results if r["train"] >= float(baseline[:args.train].mean())]
    best = max(eligible, key=lambda r: r["val"], default=None)
    report = {"target": args.target, "output": args.output, "baseline": [float(v) for v in baseline],
              "baseline_train": float(baseline[:args.train].mean()), "baseline_val": baseline_val,
              "results": [{k: v for k, v in r.items() if not k.endswith("_tensor")} for r in results]}
    if best is None or best["val"] <= baseline_val + 1e-5:
        report["decision"] = "REJECT: no validation improvement"
        Path(f"output/{args.output}.json").write_text(json.dumps(report, indent=1))
        print(report["decision"], flush=True)
        return

    scale, bias = best["scale_tensor"], best["bias_tensor"]
    sd = torch.load(f"subs/{args.target}.pt", map_location="cpu", weights_only=True)
    sd = dict(sd)
    if args.matrix:
        matrix = torch.eye(100) + best["matrix_tensor"]
        sd["head.weight"] = (matrix.T @ sd["head.weight"].float()).contiguous()
        sd["head.bias"] = (matrix.T @ sd["head.bias"].float() + bias).contiguous()
    else:
        sd["head.weight"] = (sd["head.weight"].float() * scale).contiguous()
        sd["head.bias"] = (sd["head.bias"].float() * scale + bias).contiguous()
    torch.save(sd, f"subs/{args.output}.pt")
    adjusted = dict(raw)
    for k in adjusted:
        if k != "score":
            if args.matrix:
                adjusted[k] = (adjusted[k].float() @ (torch.eye(100) + best["matrix_tensor"]) + bias).half().cpu()
            else:
                adjusted[k] = (adjusted[k].float() * scale + bias).half().cpu()
    adjusted["score"] = None
    torch.save(adjusted, D / f"{args.output}.pt")
    report["decision"] = "SAVED"
    report["selected"] = {k: v for k, v in best.items() if not k.endswith("_tensor")}
    Path(f"output/{args.output}.json").write_text(json.dumps(report, indent=1))
    print("SAVED", args.output, report["selected"], flush=True)


if __name__ == "__main__":
    main()
