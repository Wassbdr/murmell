"""Check local target assumptions against the team's completed public evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def log_softmax(values):
    centered = values - values.max(axis=-1, keepdims=True)
    return centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))


def group_outputs(models, indices, y):
    logits = np.stack([np.concatenate([parts[i] for i in indices]) for parts in models])
    labels = np.tile(y, len(indices))
    true_logp = np.take_along_axis(log_softmax(logits), labels[None, :, None], axis=2)[:, :, 0]
    ce = -true_logp
    return np.stack((ce.mean(axis=1), ce.std(axis=1)),axis=1), log_softmax(logits/2)


def gold_distribution(alternatives, labels, ce):
    base = alternatives.astype(np.float64, copy=True)
    rows = np.arange(len(labels))
    base[rows,labels] = -np.inf
    m = base.max(axis=1)
    other = m + np.log(np.exp(base-m[:,None]).sum(axis=1))
    ce = np.clip(ce, 1e-5, 20)
    base[rows,labels] = other-ce-np.log(-np.expm1(-ce))
    logp = log_softmax(base/2)
    q = np.exp(logp)
    return q.astype(np.float32), float((q*logp).sum(1).mean())


def logits_at_loss(alternatives, labels, ce):
    z=alternatives.copy()
    row=np.arange(len(labels))
    z[row,labels]=-np.inf
    maximum=z.max(1)
    logsum=maximum+np.log(np.exp(z-maximum[:,None]).sum(1))
    ce=np.clip(ce,1e-5,20.)
    z[row,labels]=logsum-ce-np.log(-np.expm1(-ce))
    return z


def rankcorr(a,b):
    def ranks(x):
        _, inverse, counts = np.unique(x,return_inverse=True,return_counts=True)
        return (np.cumsum(counts) - 0.5*(counts+1))[inverse]
    xa,xb=ranks(a),ranks(b)
    if xa.std()==0 or xb.std()==0:
        return 0.
    return float(np.corrcoef(xa,xb)[0,1])


def retention(models, orig):
    all_kl, all_agree = [], []
    for i in (3,4,5):
        old = orig[i]
        old_logp = log_softmax(old/2)
        label = np.argmax(old,axis=1)
        confidence = np.exp(log_softmax(old).max(axis=1))
        sub = np.stack([v[i] for v in models])
        kl = (np.exp(old_logp)*(old_logp-log_softmax(sub/2))).sum(2).mean(1)
        acc = np.average(sub.argmax(2) == label[None,:],weights=confidence,axis=1)
        all_kl.append(kl);all_agree.append(acc)
    kl = np.mean(all_kl,axis=0)
    agreement = np.mean(all_agree,axis=0)
    return {"agreement":agreement/(1+kl),"stable_accuracy":1/(1+kl)}


def main():
    ap=argparse.ArgumentParser()
    for arg in ("collection","raw","prior","scores","output"):
        ap.add_argument(arg,type=Path)
    ap.add_argument("--head-grid",action="store_true")
    ap.add_argument("--teacher-grid",action="store_true")
    ap.add_argument("--half",type=int,choices=(0,1))
    ap.add_argument("--record", default="codex_m70_moments64_extended")
    args=ap.parse_args()
    load=lambda p:torch.load(p,map_location="cpu",weights_only=True)
    pack = load(args.collection)
    raw, prior, official = load(args.raw),load(args.prior),json.loads(args.scores.read_text())
    selection = slice(None) if args.half is None else slice(args.half,None,2)
    y=pack["labels"].numpy()[selection];ids=pack["ids"].numpy()[selection]
    assert np.array_equal(y,raw["y"].numpy()[ids])
    def parts(tensors):return [t.float().numpy()[selection] for t in tensors]
    orig=parts(pack["original"]); models={k:parts(v) for k,v in pack["models"].items()}
    for factor in (0.8,1.5):
        models[f"w_m70a15_hs{factor}"]=[factor*v for v in models["w_m70a15"]]
    shift=(prior["za"]-raw["za"]).float().mean(0).numpy()
    counts=np.bincount(raw["y"].numpy(),minlength=100).astype(np.float32)
    frequency=counts/counts.max()
    head_settings={}
    teacher_settings={}
    if args.teacher_grid:
        for alternatives in ("lina_ES1.5","lina_ES2.0","codex_m70_moments64_extended"):
            for loss_model in ("codex_m70_moments64_extended","codex_trois_p2_l2",
                               "codex_trois_retain_prior","codex_m70_moments64","mm_55_35_wa6"):
                if alternatives==loss_model:
                    continue
                for weight in (.5,1.):
                    a,b=models[alternatives],models[loss_model]
                    key=f"teacher|{alternatives}|{loss_model}|{weight:.1f}"
                    parts_mix=[]
                    for j in range(3):
                        row=np.arange(len(y))
                        ce=-(1-weight)*log_softmax(a[j])[row,y]-weight*log_softmax(b[j])[row,y]
                        parts_mix.append(logits_at_loss(a[j],y,ce))
                    models[key]=parts_mix+a[3:]
                    teacher_settings[key]={"alternatives":alternatives,"loss_model":loss_model,
                                           "loss_weight":weight}
    if args.head_grid:
        sources=("codex_m70_moments64_extended","codex_trois_p2_l2","codex_trois_retain_prior",
                 "lina_ES1.5","codex_trois_distribution")
        grid=((1.10,1.10,0),(1.20,1.20,0),(1.25,1.0,0.4),(1.0,1.25,0.4),
              (1.16,.92,.6),(1.28,1.06,.7),(1.06,1.22,.75),(.98,1.02,1.0),
              (1.22,.84,0.),(.92,1.18,0.),(.94,.94,.5))
        for source in sources:
            if source not in models:
                continue
            for absent,removed,bias in grid:
                k=absent+(removed-absent)*frequency
                key=f"{source}|{absent:.2f}_{removed:.2f}_{bias:.2f}"
                head_settings[key]={"source":source,"absent_scale":absent,"removed_scale":removed,
                                    "bias":bias}
                models[key]=[v.mean(1,keepdims=True)+(v-v.mean(1,keepdims=True))*k+bias*shift
                             for v in models[source]]
    names=["original"]+list(models); outputs=[orig]+list(models.values())
    train=np.array([i for i,n in enumerate(names) if n in official and official[n][1]<=1834])
    check=np.array([i for i,n in enumerate(names) if n in official and official[n][1]>1834])
    assert len(train)>=6 and len(check)>=2
    observed=np.array([official.get(n,[np.nan])[0] for n in names])
    retained=retention(outputs,orig)
    by_group=[group_outputs(outputs,g,y) for g in ((0,),(1,2))]
    inputs={"prob":(2*raw["p2"].clamp_min(1e-12).log()).float().numpy()[ids],
            "logit":raw["za"].float().numpy()[ids]}
    hypotheses=[]
    for target_name, tensors in pack["loss_targets"].items():
        moments=[]; losses=[t.numpy()[selection] for t in tensors]
        for (stats,_), ce in zip(by_group,losses):
            target_stats=np.array((ce.mean(),ce.std()))
            distance=np.abs(stats-target_stats).sum(1)
            moments.append(np.clip(1-distance/max(distance[0],1e-8),0,1))
        match=np.mean(moments,axis=0)
        for base_name, base in inputs.items():
            for strength in (1.,1.5,2.25):
                for prior_weight in (0.,1.):
                    alt = strength*(base+prior_weight*shift)
                    preds=[]
                    for group, ((_, lp),ce) in enumerate(zip(by_group,losses)):
                        prob, entropy_term=gold_distribution(alt,y,ce)
                        sample_prob=prob if group==0 else np.concatenate((prob,prob))
                        div=np.maximum(entropy_term-np.einsum("kij,ij->k",lp,sample_prob)/len(sample_prob),0)
                        preds.append(np.clip(1-div/max(div[0],1e-8),0,1))
                    prediction=np.mean(preds,axis=0)
                    for keep_name,keep in retained.items():
                        proxy=match*prediction*keep
                        rho=rankcorr(proxy[train],observed[train])
                        design=np.stack([proxy[train],np.ones(len(train))],axis=1)
                        slope,intercept=np.linalg.lstsq(design,observed[train],rcond=None)[0]
                        if slope <= 0:continue
                        estimated=slope*proxy+intercept
                        mae=float(np.mean(np.abs(estimated[train]-observed[train])))
                        # Select assumptions using only receipts up to and including 1834.
                        fit=rho-2*mae
                        hypotheses.append((fit,estimated.copy(),proxy.copy(),
                            {"loss":target_name,"base":base_name,"sharpness":strength,
                             "prior":prior_weight,"retention":keep_name,"rho":rho,"mae":mae},
                            np.stack((match,prediction,keep))))
    hypotheses.sort(key=lambda h:h[0],reverse=True)
    assert hypotheses
    chosen=hypotheses[:max(1,len(hypotheses)//5)]
    estimates=np.stack([h[1] for h in chosen]); proxies=np.stack([h[2] for h in chosen])
    record=names.index(args.record)
    if args.record not in official:
        raise ValueError(f"The reference checkpoint needs an official score: {args.record}")
    terms=np.stack([h[4] for h in chosen])
    summary={}
    for i,n in enumerate(names[1:],1):
        summary[n]={"historical_score":official.get(n,[None])[0],
                    "median_calibrated":float(np.median(estimates[:,i])),
                    "span_calibrated":np.quantile(estimates[:,i],[.1,.9]).tolist(),
                    "fraction_ahead_of_record":float((proxies[:,i]>proxies[:,record]).mean()),
                    "anchored_estimate":float(observed[record]+np.median(estimates[:,i]-estimates[:,record])),
                    "median_changes":[float(v) for v in np.median(terms[:,:,i]-terms[:,:,record],axis=0)],
                    "all_three_non_decreasing":float(np.all(terms[:,:,i]>=terms[:,:,record]-1e-5,axis=1).mean())}
    best_flags=[h[3] for h in chosen]
    check_mae=float(np.abs(np.median(estimates[:,check],axis=0)-observed[check]).mean())
    result={"historical_fit":{names[i]:float(observed[i]) for i in train},
            "later_receipts":{names[i]:float(observed[i]) for i in check},
            "record":args.record,"later_mae":check_mae,"variants":len(hypotheses),"selected":len(chosen),
            "selection_rule":"Top fifth by historical rank correlation minus twice absolute calibration error through receipt 1834. The later receipts are retrospective checks.",
            "assumptions":best_flags,"models":summary,"head_settings":head_settings,
            "synthetic_teachers":teacher_settings}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("Historical scores",len(train),"later",len(check),"later MAE",round(check_mae,4))
    for n,m in sorted(summary.items(),key=lambda v:v[1]["median_calibrated"],reverse=True)[:12]:
        print(n, round(m["median_calibrated"],3),round(m["fraction_ahead_of_record"],2),m["historical_score"])
    print("Best target assumption",best_flags[0])


if __name__=="__main__":
    main()
