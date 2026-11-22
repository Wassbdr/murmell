"""CNSU phase A. One conservative, measured SGD update to the official 4112 state."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def options():
    p = argparse.ArgumentParser()
    p.add_argument("--home", type=Path, default=Path.home() / "t3")
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--kimi", type=Path)
    p.add_argument("--moments", action="store_true")
    p.add_argument("--multi", action="store_true")
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--save-controls", action="store_true")
    p.add_argument("--train-sources", type=int, default=256)
    p.add_argument("--dev-sources", type=int, default=350)
    p.add_argument("--step-norm", type=float, default=.24)
    return p.parse_args()


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def probability(logits):
    return F.softmax(logits.float() / 2, dim=1)


def kl(logits, target):
    return F.kl_div(F.log_softmax(logits.float() / 2, 1), target.float(), reduction="none").sum(1)


def wrong_class_kl(logits, target, labels):
    mask = F.one_hot(labels, num_classes=logits.shape[1]).bool()
    q = target.masked_fill(mask,0)
    q = q/q.sum(1,keepdim=True).clamp_min(1e-12)
    return kl(logits.masked_fill(mask,-1e4),q)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = options()
    start = time.monotonic()
    args.out.mkdir(exist_ok=True, parents=True)
    sys.path.insert(0, str(args.home))
    import task_template as tt
    from t3_consist import alter
    torch.set_num_threads(4)
    torch.manual_seed(164011)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    original = tt.build_model().cuda().eval().requires_grad_(False)
    original.load_state_dict(load_state(args.home / "model.pt"), strict=True)
    student = tt.build_model().cuda().eval().requires_grad_(False)
    student.load_state_dict(load_state(args.anchor), strict=True)
    selected_names = [name for name, _ in student.named_parameters()
                      if name.startswith(("blocks.10.", "blocks.11.", "head."))]
    params = [p.requires_grad_(True) for n, p in student.named_parameters() if n in selected_names]
    nparam = sum(p.numel() for p in params)
    def flat(x):
        return torch.cat([v.reshape(-1) for v in x])
    initial = flat([p.detach() for p in params])
    def step(direction, size):
        with torch.no_grad():
            vector = initial if direction is None else initial - size * direction
            offset = 0
            for p in params:
                end = offset + p.numel()
                p.copy_(vector[offset:end].view_as(p))
                offset = end
    def prefix(x):
        z = student.patch_embed(x)
        z = student._pos_embed(z)
        z = student.patch_drop(z)
        z = student.norm_pre(z)
        for block in student.blocks[:-2]:
            z = block(z)
        return z
    def tail(z):
        with torch.autocast("cuda", dtype=torch.float16):
            for block in student.blocks[-2:]:
                z = block(z)
            return student.forward_head(student.norm(z)).float()
    target = load_state(args.target)
    p2 = target["p2"].float()
    p2 /= p2.sum(1, keepdim=True)
    pseudo = target["y"].long()
    kimi = None
    if args.kimi:
        artifact = load_state(args.kimi)
        assert torch.equal(artifact["y"].long(),pseudo)
        kimi = artifact["pg"].float()
        kimi = kimi / kimi.sum(1,keepdim=True)
    secondary = {}
    original_metrics = {}
    data = tt.load_forget_images(args.home / "forget_images.npz")
    synth = np.load(args.home / "gen/gen_test.npz")
    sim = torch.from_numpy(synth["images"][:650].copy())
    syl = torch.from_numpy(synth["cls"][:650].copy()).long()
    gen = torch.Generator().manual_seed(79623)
    train_ids = torch.randperm(2500, generator=gen)[:args.train_sources]
    dev_ids = 2500 + torch.randperm(500, generator=gen)[:args.dev_sources]
    assert train_ids.max() < 2500 and dev_ids.min() >= 2500 and dev_ids.max() < 3000
    prepared = {}
    discrepancy = 0.
    def prepare(key, pixels, ids, labels, view=False, seed=0):
        nonlocal discrepancy
        g = torch.Generator(device="cuda").manual_seed(seed)
        tokens, targets, categories = [], [], []
        alt_replaced = 0
        secondary_targets = []
        orig_logits = []
        with torch.no_grad():
            for ix in ids.split(24):
                x = pixels[ix].to("cuda").permute(0,3,1,2).float() / 255
                if view:
                    x = alter(x, g, .7)
                x = 2*x-1
                with torch.autocast("cuda", dtype=torch.float16):
                    h = prefix(x)
                    zo = original(x).float()
                    if not tokens and key == "train_exact":
                        discrepancy = float((tail(h)-student(x).float()).abs().max())
                        assert discrepancy < .01, discrepancy
                lab = labels[ix].cuda()
                orig_logits.append(zo)
                q = probability(zo)
                if pixels is data:
                    qbase = p2[ix].cuda()
                    if kimi is not None:
                        q2 = kimi[ix].cuda()
                        # A separate proxy estimator, frozen before the experiment.
                        # Use its image distribution as stored on both views.
                        secondary_targets.append(q2)
                    if view:
                        ce_o = F.cross_entropy(zo, lab, reduction="none")
                        ce_ref = F.cross_entropy(2*qbase.clamp_min(1e-12).log(), lab, reduction="none")
                        replace = ce_o > ce_ref
                        q = torch.where(replace[:,None], q, qbase)
                        alt_replaced += int(replace.sum())
                    else:
                        q = qbase
                tokens.append(h.detach())
                targets.append(q)
                categories.append(lab)
        prepared[key] = tuple(torch.cat(t) for t in (tokens, targets, categories))
        if secondary_targets:
            secondary[key] = torch.cat(secondary_targets)
        z = torch.cat(orig_logits)
        _,target_q,yl = prepared[key]
        valid = yl >= 0
        ce = F.cross_entropy(z[valid],yl[valid],reduction="none")
        original_metrics[key] = dict(kl=kl(z,target_q).mean().item(),ce=ce.mean().item(),
                                     ce_std=ce.std(unbiased=False).item(),
                                     agreement=(z.argmax(1)==target_q.argmax(1)).float().mean().item(),
                                     accuracy=(z[valid].argmax(1)==yl[valid]).float().mean().item(),
                                     n_labeled=int(valid.sum()),n=len(yl))
        if key in secondary:
            original_metrics[key]["kimi_kl"] = kl(z,secondary[key]).mean().item()
        return alt_replaced
    repl = {}
    for tag, ids, seed in [("train",train_ids,64201),("dev",dev_ids,64202)]:
        prepare(tag+"_exact",data,ids,pseudo)
        repl[tag] = prepare(tag+"_alt",data,ids,pseudo,True,seed)
    for tag, ids in [("basis",torch.arange(256)),("trust",torch.arange(256,384)),("val",torch.arange(384,650))]:
        prepare(tag,sim,ids,syl)
    del original, data, sim, synth
    torch.cuda.empty_cache()
    print("prepared sources",round(time.monotonic()-start,2),"s",flush=True)

    def accumulate(groups, secondary_teacher=False, conditional=False):
        total = torch.zeros_like(initial)
        for name, weight in groups:
            x,q,y_fit = prepared[name]
            if secondary_teacher:
                q = secondary[name]
            for indexes in torch.arange(len(q),device="cuda").split(16):
                z_fit = tail(x[indexes])
                losses = (wrong_class_kl(z_fit,q[indexes],y_fit[indexes]) if conditional else kl(z_fit,q[indexes]))
                loss = losses.sum() / len(q) * weight * 4
                total.add_(flat(torch.autograd.grad(loss,params)).detach())
        return total
    gf = accumulate([("train_exact",.5),("train_alt",.5)])
    gr = accumulate([("basis",1.)])
    # Independent Gaussian score probes define true categorical-logit VJPs.
    # Each row uses positives and negatives from all 100 classes. The categorical
    # covariance is computed at the original model, the Jacobian at the anchor.
    x,q,_ = prepared["basis"]
    banks = torch.empty((args.rank,nparam),device="cuda",dtype=torch.float32)
    probe_rng = torch.Generator(device="cuda").manual_seed(29076)
    base_order = torch.randperm(len(q), generator=gen).cuda()
    for j in range(args.rank):
        positions = torch.arange(j*8,j*8+8,device="cuda") % len(q)
        idx = base_order[positions]
        p = q[idx]
        noise = torch.randn(p.shape,generator=probe_rng,device="cuda")
        v = p.sqrt()*noise
        v -= p*v.sum(1,keepdim=True)
        objective = ((tail(x[idx])/2)*v).sum() / math.sqrt(len(idx))
        row = flat(torch.autograd.grad(objective,params)).detach()
        banks[j].copy_(row / row.norm().clamp_min(1e-10))
    gram = banks @ banks.T
    eig, vec = torch.linalg.eigh(gram.double())
    keep = eig > eig.max()*1e-7
    coeff = vec[:,keep] @ ((vec[:,keep].T @ (banks @ gf).double())/eig[keep])
    proj = banks.T @ coeff.float()
    residual_projection = (banks @ (gf-proj)).norm() / (banks @ gf).norm()
    dot_frac = (gf@proj/gf.square().sum()).item()
    diagnostics = dict(rank_requested=args.rank, rank_effective=int(keep.sum()),
                       min_eigenvalue=eig.min().item(), max_eigenvalue=eig.max().item(),
                       parameter_count=nparam, parameter_groups=selected_names,
                       grad_forget_norm=gf.norm().item(), grad_retain_norm=gr.norm().item(),
                       removed_gradient_energy_fraction=dot_frac,
                       reprojection_residual_fraction=residual_projection.item(),
                       prefix_logit_discrepancy=discrepancy,
                       alt_replaced=repl, device=torch.cuda.get_device_name(),
                       torch_version=torch.__version__)
    if args.moments:
        # Gradients of four local empirical moments, measured only on fit sources.
        moments = []
        for name in ("train_exact", "train_alt"):
            xx,_,yy = prepared[name]
            with torch.no_grad():
                ll = torch.cat([F.cross_entropy(tail(a),b,reduction="none")
                                for a,b in zip(xx.split(16),yy.split(16))])
            for weights in (torch.ones_like(ll),(ll-ll.mean())/ll.std(unbiased=False).clamp_min(1e-7)):
                grad = torch.zeros_like(gf)
                for a,b,w in zip(xx.split(16),yy.split(16),weights.split(16)):
                    objective = (w*F.cross_entropy(tail(a),b,reduction="none")).sum()/len(yy)
                    grad.add_(flat(torch.autograd.grad(objective, params)).detach())
                moments.append(grad/grad.norm().clamp_min(1e-10))
        constrained = torch.stack(moments)
        union = torch.cat((banks,constrained))
        def null_direction(bank, direction):
            gram = (bank @ bank.T).double()
            values,vectors = torch.linalg.eigh(gram)
            ok = values > values.max()*1e-7
            d = vectors[:,ok] @ ((vectors[:,ok].T @ (bank @ direction).double()) / values[ok])
            return direction-bank.T @ d.float()
        add_directions = {"moments_only":null_direction(constrained,gf),
                          "projected_moments":null_direction(union,gf),
                          "kl_only_moments":null_direction(constrained,gf+gr),
                          "projected_kl_moments":null_direction(union,gf+gr)}
        if args.multi:
            assert kimi is not None
            # Compute each source/view loss gradient separately on FIT only.
            joint = []
            for aux in (False,True):
                for v in ("train_exact","train_alt"):
                    joint.append(null_direction(union,accumulate([(v,1.)],aux)))
            joint.append(null_direction(union,gr))
            unprojected_rows = []
            for aux in (False,True):
                for v in ("train_exact","train_alt"):
                    unprojected_rows.append(null_direction(constrained,accumulate([(v,1.)],aux)))
            unprojected_rows.append(null_direction(constrained,gr))
            def common(gradient_list):
                unit = torch.stack([g/g.norm().clamp_min(1e-12) for g in gradient_list])
                gram = (unit@unit.T).double().cpu()
                # Frank-Wolfe minimum-norm point in convex hull of unit gradients.
                w = torch.full((len(gram),),1/len(gram),dtype=torch.float64)
                for _ in range(500):
                    products = gram@w
                    j = int(products.argmin())
                    move = -w.clone()
                    move[j] += 1
                    gap = -float(move@products)
                    if gap < 1e-8:
                        break
                    eta = min(1., max(0.,gap/max(float(move@gram@move),1e-12)))
                    w += eta*move
                return w.float().cuda()@unit, w.tolist(),(gram@w).tolist()
            for name,gs in [("projected_common4",joint[:4]),("projected_common5",joint),
                            ("moments_common4",unprojected_rows[:4]),("moments_common5",unprojected_rows)]:
                add_directions[name],ws,dots = common(gs)
                diagnostics[name] = {"fit_weights":ws,"fit_inner_products":dots}
            if args.conditional:
                for tag,bank in (("projected_conditional5",union),("moments_conditional5",constrained)):
                    independent = [null_direction(bank,accumulate([(v,1.)],aux,True))
                                   for aux in (False,True) for v in ("train_exact","train_alt")]
                    independent.append(null_direction(bank,gr))
                    add_directions[tag],ws,dots = common(independent)
                    diagnostics[tag] = {"fit_weights":ws,"fit_inner_products":dots}
            del joint,unprojected_rows
        del union,constrained,moments
    else:
        add_directions = {}
    del banks
    print("basis",diagnostics["rank_effective"],"energy",dot_frac,round(time.monotonic()-start,2),flush=True)

    @torch.no_grad()
    def metrics(name):
        x,q,y = prepared[name]
        z = torch.cat([tail(part) for part in x.split(32)])
        valid = y >= 0
        ce = F.cross_entropy(z[valid],y[valid],reduction="none")
        return dict(kl=kl(z,q).mean().item(),
                    kimi_kl=kl(z,secondary[name]).mean().item() if name in secondary else None,
                    wrong_p2=wrong_class_kl(z,q,y).mean().item() if name in secondary else None,
                    wrong_kimi=wrong_class_kl(z,secondary[name],y).mean().item() if name in secondary else None,
                    n_labeled=int(valid.sum()),
                    accuracy=(z[valid].argmax(1)==y[valid]).float().mean().item(),
                    agreement=(z.argmax(1)==q.argmax(1)).float().mean().item(),
                    ce=ce.mean().item(),ce_std=ce.std(unbiased=False).item(),n=len(y))
    def evaluate():
        return {tag: metrics(tag) for tag in ["trust","val","dev_exact","dev_alt"]}
    base = evaluate()
    maximum_trust_kl = base["trust"]["kl"] + max(.001, .01*base["trust"]["kl"])
    minimum_trust_acc = base["trust"]["accuracy"] - 2/base["trust"]["n_labeled"]
    minimum_trust_agreement = base["trust"]["agreement"] - 2/base["trust"]["n"]
    directions = {"unconstrained":gf, "kl_only":gf + gr}
    for strength in (.5,.9,1.):
        directions["projected_"+str(strength)] = gf - strength*proj
    directions.update(add_directions)
    rows = {"unchanged":dict(norm=0.,metrics=base,trace=[])}
    best_score = -float("inf")
    best_name = None
    best_vector = None
    for name,direction in directions.items():
        # SGD on the effective direction, global unit norm clipping, no momentum.
        direction = direction / direction.norm().clamp_min(1e-12)
        traces = []
        for attempt in range(7):
            size = args.step_norm*.5**attempt
            step(direction,size)
            tm = metrics("trust")
            passed = (tm["kl"] <= maximum_trust_kl and tm["accuracy"] >= minimum_trust_acc
                      and tm["agreement"] >= minimum_trust_agreement)
            traces.append(dict(norm=size,trust=tm,passed=passed))
            if passed:
                break
        if not passed:
            step(None,0)
            size = 0.
        result = evaluate()
        gains = [base[t]["kl"]-result[t]["kl"] for t in ("dev_exact","dev_alt")]
        retain_cost = result["val"]["kl"] - base["val"]["kl"]
        ratio_gain = sum(gains[i]/max(base[t]["kl"],1e-7) for i,t in enumerate(("dev_exact","dev_alt")))/2
        score = ratio_gain - max(0.,retain_cost)/(1+base["val"]["kl"])
        moment_drift = {t: [result[t]["ce"] - base[t]["ce"], result[t]["ce_std"] - base[t]["ce_std"]] for t in ("dev_exact","dev_alt")}
        eligible = (min(gains) > 0 and retain_cost <= max(.001,.01*base["val"]["kl"])
                    and result["val"]["accuracy"] >= base["val"]["accuracy"]-2/base["val"]["n_labeled"]
                    and result["val"]["agreement"] >= base["val"]["agreement"]-2/base["val"]["n"])
        kimi_gains = None
        if kimi is not None:
            kimi_gains = [base[t]["kimi_kl"] - result[t]["kimi_kl"] for t in ("dev_exact","dev_alt")]
            score += sum(kimi_gains[i]/max(base[t]["kimi_kl"],1e-7) for i,t in enumerate(("dev_exact","dev_alt")))/2
        if args.moments:
            max_moment = max(abs(v) for pair in moment_drift.values() for v in pair)
            eligible = eligible and max_moment <= .035 and kimi_gains is not None and min(kimi_gains) >= -.001
            score -= max_moment / max(base["dev_exact"]["ce"],1e-7)
        rows[name] = dict(norm=size, kimi_gains=kimi_gains, metrics=result, trace=traces,
                          gains_forget_kl=gains, retain_kl_cost=retain_cost, ce_moment_drifts=moment_drift, eligible=eligible, score=score)
        if name.startswith("projected") and (not args.conditional or name == "projected_conditional5") and eligible and score > best_score:
            best_score = score
            best_name = name
            best_vector = (direction.clone(),size)
        print(name,size,[round(v,7) for v in gains],"ret",round(retain_cost,7),"eligible",eligible,flush=True)
    if best_name is None:
        # Export the best actual projected update even when the decision is reject.
        best_name = ("projected_conditional5" if args.conditional else max((k for k in rows if k.startswith("projected")),key=lambda n:rows[n]["score"]))
        d = directions[best_name]
        best_vector = d/d.norm().clamp_min(1e-12),rows[best_name]["norm"]
    if args.save_controls:
        for key in ("unconstrained","kl_only","moments_only","kl_only_moments","moments_common4","moments_common5","projected_common4","projected_common5","projected_conditional5","moments_conditional5"):
            if key not in directions:
                continue
            vec = directions[key]
            step(vec/vec.norm().clamp_min(1e-12),rows[key]["norm"])
            weights = {k:v.detach().float().cpu().contiguous() for k,v in student.state_dict().items()}
            torch.save(weights,args.out/(key+".pt"))
            del weights
    step(*best_vector)
    candidate = args.out / "cnsu_phase_a_candidate.pt"
    state = {n:v.detach().float().cpu().contiguous() for n,v in student.state_dict().items()}
    assert len(state)==176 and all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values())
    torch.save(state,candidate)
    tt.build_model().load_state_dict(load_state(candidate), strict=True)
    assert candidate.stat().st_size < 450_000_000
    diagnostics.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                       runtime_seconds=time.monotonic()-start)
    result = dict(method="CNSU A",anchor_sha256=digest(args.anchor),target_sha256=digest(args.target),
                  args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  diagnostics=diagnostics, original_metrics=original_metrics,rows=rows,best_projected=best_name,
                  candidate_sha256=digest(candidate),candidate_size=candidate.stat().st_size,
                  candidate_passes_dev=rows[best_name]["eligible"],
                  fit_ids=train_ids.tolist(), dev_ids=dev_ids.tolist(),
                  trust_limits=dict(max_kl=maximum_trust_kl,min_acc=minimum_trust_acc,
                                    min_agree=minimum_trust_agreement))
    (args.out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print("finished",best_name,result["candidate_sha256"],flush=True)


if __name__ == "__main__":
    main()
