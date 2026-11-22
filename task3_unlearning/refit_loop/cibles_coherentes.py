"""Find one classifier distribution compatible with the ensemble KL and loss moments."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def loss_values(true, concentration):
    bounds = torch.finfo(true.dtype)
    return F.softplus(concentration - 2*torch.logit(true.clamp(bounds.tiny, 1-bounds.eps)))


def calibrated_losses(log_shape, mean, sd):
    # A positive power preserves order. Its coefficient of variation increases
    # monotonically, so a scalar search enforces both required moments.
    centered = log_shape - log_shape.max()
    target = sd / mean
    def scale(exponent):
        t = (exponent * centered).softmax(0) * len(centered)
        return t, t.std()
    lo, hi = 0., 1.
    for _ in range(16):
        if scale(hi)[1] >= target:
            break
        hi *= 2
    for _ in range(30):
        mid = .5*(lo+hi)
        if scale(mid)[1] < target:
            lo = mid
        else:
            hi = mid
    return mean * scale(.5*(lo+hi))[0]


def solve_distribution(probability, labels, wanted):
    q = probability.double()
    rows = torch.arange(len(q))
    true = q[rows,labels].clamp(1e-8,1-1e-8)
    wrong = q.clone()
    wrong[rows,labels] = 0
    wrong /= wrong.sum(1,keepdim=True).clamp_min(1e-15)
    concentration = wrong.square().sum(1).log()
    mean,sd = wanted.double()
    initial = loss_values(true,concentration).clamp_min(1e-8)
    def log_odds(ce):
        return .5*(concentration - ce - (-torch.expm1(-ce)).clamp_min(1e-20).log())
    def cross_entropy(ce):
        return F.binary_cross_entropy_with_logits(log_odds(ce),true)
    ce = calibrated_losses(initial.log(),mean,sd)
    start = float(cross_entropy(ce))
    rate = 2.
    for _ in range(75):
        pi = log_odds(ce).sigmoid()
        grad = -.5*(pi-true)*ce/(-torch.expm1(-ce)).clamp_min(1e-20)
        tangent1 = F.normalize(ce, dim=0)
        remaining = ce.square()-torch.dot(ce.square(),tangent1)*tangent1
        tangent2 = F.normalize(remaining, dim=0)
        grad -= torch.dot(grad,tangent1)*tangent1 + torch.dot(grad,tangent2)*tangent2
        old = cross_entropy(ce)
        proposal = calibrated_losses(ce.clamp_min(1e-14).log()-rate*grad,mean,sd)
        if cross_entropy(proposal) <= old:
            ce = proposal
            rate = min(4.,rate*1.10)
        else:
            rate *= .5
        if rate < .001 or grad.abs().max() < 1e-6:
            break
    pi = log_odds(ce).sigmoid()
    output = (1-pi[:,None])*wrong
    output[rows,labels] = pi
    actual = loss_values(pi,concentration)
    if not torch.allclose(torch.stack((actual.mean(),actual.std())),wanted.double(),atol=.003,rtol=.001):
        raise ValueError("The requested gold moments could not be reproduced")
    info = {"requested":wanted.tolist(),"attained":[float(actual.mean()),float(actual.std())],
            "binary_cross_entropy_before":start,"binary_cross_entropy_after":float(cross_entropy(ce))}
    return output.float(),info


def ideal_forget(predictions, members, altered_members, moments, originals, y):
    rows=torch.arange(len(y))
    values=[]
    def error(prob,target):
        p=prob.double().square()
        ce=(p.sum(1).log()-p[rows,y].clamp_min(1e-30).log())
        return (torch.stack((ce.mean(),ce.std()))-target).abs().sum()
    def divergence(q,p):
        return (q*(q.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum(1).mean()
    for q,a,goal in zip(members,altered_members,moments):
        losses=[];scores=[]
        for view, (ref, out, original) in enumerate(zip((q,a),predictions,originals)):
            target=goal[2*view:2*view+2]
            losses.append((1-error(out,target)/error(original,target).clamp_min(1e-7)).clamp(0,1))
            scores.append((1-divergence(ref,out)/divergence(ref,original).clamp_min(1e-7)).clamp(0,1))
        values.append(torch.stack(losses).mean()*torch.stack(scores).mean())
    return float(torch.stack(values).quantile(.25))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("gold",type=Path);p.add_argument("output",type=Path);p.add_argument("report",type=Path)
    p.add_argument("--original",type=Path,required=True);p.add_argument("--raw-cache",action="store_true")
    args=p.parse_args()
    gold=torch.load(args.gold,weights_only=True,map_location="cpu")
    raw=torch.load(args.original,weights_only=True,map_location="cpu")
    y=gold["y"].long();q=gold["members"];a=gold.get("altered_members",q);ms=gold["individual_moments"]
    if args.raw_cache:
        if not torch.equal(y,raw["y"].long()):
            raise ValueError("Original and gold image ordering differs")
        original=[(raw[k].float()/2).softmax(1) for k in ("zc","za")]
    else:
        original=[(raw[k].float()/2).softmax(1) for k in ("clean","aug60")]
    folds=[torch.arange(i,len(q),2) for i in (0,1)]
    trials=[];solutions={}
    grid=(0.,.25,.5,.75,1.)
    for fit,held in (folds,folds[::-1]):
        centers=[q[fit].mean(0),a[fit].mean(0)]
        target=ms[fit].median(0).values
        opt=[solve_distribution(m,y,target[2*i:2*i+2])[0] for i,m in enumerate(centers)]
        if int(fit[0])==0:
            solutions["fit"]=centers,opt
        trials.append([ideal_forget([torch.lerp(c,o,s) for c,o in zip(centers,opt)],
                                    q[held],a[held],ms[held],original,y) for s in grid])
    # Select the largest worst-fold improvement rather than favouring one teacher set.
    gains=[[s-base[0] for s in base] for base in trials]
    winner=max(range(len(grid)),key=lambda i:min(g[i] for g in gains))
    weight=grid[winner]
    centers=[q.mean(0),a.mean(0)];target=ms.median(0).values
    all_solved=[solve_distribution(c,y,target[2*i:2*i+2]) for i,c in enumerate(centers)]
    solutions["full"]=centers,[part[0] for part in all_solved]
    output={"y":y,"strength":weight,"source_moments":ms.clone()}
    for name,(base,opt) in solutions.items():
        output[name]=torch.stack([torch.lerp(c,o,weight) for c,o in zip(base,opt)])
    info={"strength":weight,"grid":grid,"heldout_quality":trials,"gain_over_raw":gains,
          "full_constraints":[part[1] for part in all_solved],"images":len(y)}
    tmp=args.output.with_suffix(".partial");torch.save(output,tmp);tmp.replace(args.output)
    args.report.write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info),flush=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    main()
