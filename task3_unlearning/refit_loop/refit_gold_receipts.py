"""Update calibrated targets from official score differences using bounded local steps."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from calibrated_gold import adjust_classes, altered_probabilities, class_features, projection_delta, rank, shadow_distribution, shadow_targets


def transfer_weights(frozen, current, parameter):
    by_config = {}
    for fit in frozen:
        by_config.setdefault(tuple(fit["cfg"]), []).append(fit[parameter])
    result = []
    for fit in current:
        observed = by_config.get(tuple(fit["cfg"]), [0.])
        weight = float(torch.tensor(observed).median())
        result.append(dict(fit, **{parameter: weight}))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--through",type=int,help="Last official receipt to include")
    parser.add_argument("--require-receipt", action="store_true", help="Require the exact final receipt and measured features")
    parser.add_argument("--class-features",action="store_true")
    parser.add_argument("--projection-cache", type=Path)
    parser.add_argument("--projection-key", default="delta_35")
    parser.add_argument("--separate-altered", action="store_true")
    parser.add_argument("--shadow-cache", type=Path)
    parser.add_argument("--shadow-mode", choices=("raw","rank"), default="raw")
    parser.add_argument("--response-cache", type=Path)
    parser.add_argument("--response-label", choices=("reponses","familles","details","voisins"), default="reponses")
    parser.add_argument("--transfer-fit", type=Path)
    parser.add_argument("--base-through",type=int,help="Use the fit frozen at this receipt")
    parser.add_argument("--forecast",nargs="+",default=[])
    parser.add_argument("--forecast-output",type=Path)
    parser.add_argument("--forecast-fit", type=Path)
    args=parser.parse_args()
    if args.require_receipt and args.through is None:
        parser.error("--require-receipt requires --through")
    if args.forecast_fit and not args.forecast:
        parser.error("--forecast-fit requires model names to forecast")
    temporal = args.projection_cache or args.separate_altered or args.shadow_cache or args.response_cache
    if temporal and not args.base_through:
        parser.error("Use a frozen base fit for an independent comparison")
    if sum(map(bool, [args.projection_cache, args.separate_altered, args.shadow_cache, args.class_features, args.response_cache])) > 1:
        parser.error("Validate one change to the gold targets at a time")
    if args.transfer_fit and not temporal:
        parser.error("--transfer-fit applies to a validated change in gold targets")
    root=Path.cwd();base=root/"cache/lina_feat";ours=root/"cache/codex_prevision_privee/cache/lina_feat"
    def load(path):return torch.load(path,map_location="cpu",weights_only=True)
    meta,orig,configs=load(base/"_meta.pt"),load(base/"_orig.pt"),load(base/"_fit.pt")
    if args.base_through:
        configs=load(root/"cache"/f"codex_fit_recu_{args.base_through}.pt")
    if args.forecast_fit:
        configs=load(args.forecast_fit)
    recipient_configs = load(args.transfer_fit) if args.transfer_fit else configs
    y=meta["y"].long()
    sample=torch.randperm(len(y),generator=torch.Generator().manual_seed(1789))[:1024]
    projected = projection_delta(args.projection_cache, args.projection_key, y)[sample] if args.projection_cache else None
    shadows = [q[sample] for q in shadow_targets(args.shadow_cache, y)] if args.shadow_cache else None
    if args.response_cache:
        from directions_gold import tilt
        response_pack = load(args.response_cache)
        if not torch.equal(response_pack["y"], y):
            raise ValueError("Model responses use different image order")
        response_basis = {k: v[:,sample].float() for k,v in response_pack.items()
                          if isinstance(v,torch.Tensor) and v.ndim == 3}
    known={}
    for spec in sorted((root/"tools").glob("lina_spec*.txt")):
        for line in spec.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            n,_,_,_,v=line.split("|")
            file=base/(n+".pt")
            if v=="?" or not file.is_file():
                continue
            score=float(v)
            if not 0<=score<=1:
                continue
            if n in known and abs(known[n][1]-score)>.00011:
                raise ValueError(f"Inconsistent team score for {n} in {spec.name}")
            known[n]=[file,score,0]
    receipts=[]
    for file in (root/"output/receipts").glob("*.json"):
        a=json.loads(file.read_text());n=Path(a.get("model","")).stem
        if args.through and a.get("submission_id",0)>args.through:
            continue
        src=next((d/(n+".pt") for d in (ours,root/"cache/codex_controle_familles",base) if (d/(n+".pt")).is_file()),None)
        if src and a.get("status")=="success":
            known[n]=[src,float(a["score"]),int(a["submission_id"])]
            if a["submission_id"]>=2047:receipts.append((a["submission_id"],n))
    receipts.sort()
    if args.require_receipt and (not receipts or receipts[-1][0] != args.through):
        raise ValueError(f"Missing official receipt or measured features for {args.through}")
    assert len(receipts)>=2
    names=list(known);champion=names.index("lina_ES2.0");last=names.index(receipts[-1][1])
    actual=torch.tensor([known[n][1] for n in names],dtype=torch.float64)
    base_weights=torch.tensor([3. if known[n][2]>=2047 else 1. for n in names],dtype=torch.float64)
    fams=list(dict.fromkeys(["clean"]+[c["cfg"][0] for c in configs]))
    def prep(f):
        means=[];stds=[];lps=[]
        for fam in fams:
            z=f[fam].float();loss=F.cross_entropy(z,y,reduction="none")
            means.append(loss.mean());stds.append(loss.std());lps.append((z[sample]/2).log_softmax(1))
        keep=[];ag=[];high=[]
        for kind in meta["kinds"]:
            o=orig["anc_"+kind].float();z=f["anc_"+kind].float()
            a=(o/2).log_softmax(1);b=(z/2).log_softmax(1);p=a.exp()
            d=(p*(a-b)).sum(1);keep.append(d.mean());ag.append((z.argmax(1)==o.argmax(1)).float().mean())
            mask=p.max(1).values>.9;high.append(d[mask].mean() if mask.any() else d.mean())
        return (torch.stack(means),torch.stack(stds),torch.stack(lps),torch.stack(keep).mean(),
                torch.stack(ag).mean(),torch.stack(high).mean())
    featured=[prep(orig)]+[prep(load(known[n][0])) for n in names]
    all_names=names.copy()
    for name in args.forecast:
        if name in all_names:
            continue
        paths=(ours,root/"cache/codex_controle_familles",base)
        source=next((d/(name+".pt") for d in paths if (d/(name+".pt")).is_file()),None)
        if source is not None:
            all_names.append(name)
            featured.append(prep(load(source)))
    mu,std,lp,kl,agree,kh=map(torch.stack,zip(*featured))
    geo=[0,1,2,3,10];nm=orig["nonmem"].float()
    loss_nm=torch.stack([F.cross_entropy(nm[:,i],y,reduction="none") for i in range(nm.shape[1])],1)
    removed,stability,classes=class_features(y,loss_nm[:,geo].mean(1),meta["diff"].float())
    class_covariates=removed[sample],stability[sample],classes
    ranks={"aug":rank(meta["diff"].float())[sample],"nmg":rank(loss_nm[:,geo].mean(1))[sample],
           "nmall":rank(loss_nm.mean(1))[sample]}
    bases={"za":meta["za"][sample].float(),"nmg":nm[sample][:,geo].mean(1),
           "clean":orig["clean"][sample].float(),"p2za":2*meta["p2"][sample].float().clamp_min(1e-9).log()}
    mask=F.one_hot(y[sample],100).bool()
    scale=torch.tensor([.6,.5,.6,.5,.24,.35,.35,.25,.3,.3,.3])
    def simulate(t,cfg,extra=None,projection_weight=0,altered_weight=0,shadow_weight=0,response=None):
        fam,src,r=cfg;i=fams.index(fam);t=t.float()
        target=F.softplus(t[:4]);means=target[[0,2]];sds=target[[1,3]]
        dist=(mu[:,[0,i]]-means).abs()+(std[:,[0,i]]-sds).abs()
        m=(1-dist[1:]/dist[:1].clamp_min(1e-6)).clamp(0,1).mean(1)
        pi=t[5].sigmoid()+(t[6].sigmoid()-t[5].sigmoid())*torch.sigmoid((t[7].sigmoid()-ranks[r])/.04)
        base_logits = bases[src]
        if projection_weight:
            base_logits = base_logits + projection_weight * projected
        logits=base_logits/(2*t[4].exp())
        if extra is not None:
            pi,logits=adjust_classes(pi,logits,class_covariates,extra)
        q=(1-pi[:,None])*logits.masked_fill(mask,-1e4).softmax(1)+pi[:,None]*mask
        q_altered = altered_probabilities(q, y[sample], target, altered_weight) if altered_weight else q
        if shadow_weight:
            q_altered = torch.lerp(q_altered, shadow_distribution(q_altered,shadows[1],y[sample],args.shadow_mode), shadow_weight)
            q = torch.lerp(q, shadow_distribution(q,shadows[0],y[sample],args.shadow_mode), shadow_weight)
        if response is not None:
            q_altered = tilt(q_altered, response_basis[fam], response)
            q = tilt(q, response_basis["clean"], response)
        negentropy=(q*q.clamp_min(1e-12).log()).sum(1).mean()
        div=negentropy-torch.einsum("nc,mgnc->mg",q,lp[:,[0,i]])/len(q)
        if altered_weight or shadow_weight or response is not None:
            entropy = (q_altered * q_altered.clamp_min(1e-12).log()).sum(1).mean()
            div[:,1] = entropy - torch.einsum("nc,mnc->m", q_altered, lp[:,i]) / len(q)
        pred=(1-div[1:]/div[:1].clamp_min(1e-6)).clamp(0,1).mean(1)
        keep=agree[1:].pow(F.softplus(t[10]))/(1+F.softplus(t[8])*kl[1:]+F.softplus(t[9])*kh[1:])
        answer=(m*pred*keep).double()
        return answer+actual[champion]-answer[champion]

    if args.forecast:
        best=int(actual.argmax())
        forecasts = []
        for c in configs:
            opt = {}
            if any(c.get("response_coefficients", [])):
                if not args.response_cache:
                    parser.error("The frozen fit requires --response-cache")
                opt["response"] = torch.tensor(c["response_coefficients"])
            if c.get("shadow_weight", 0):
                if not args.shadow_cache or args.shadow_mode != c.get("shadow_mode","raw"):
                    parser.error("The frozen fit requires its auxiliary probabilities and matching mode")
                opt["shadow_weight"] = c["shadow_weight"]
            if c.get("projection_weight", 0):
                if not args.projection_cache or args.projection_key != c["projection_key"]:
                    parser.error("The frozen fit requires its projection-cache and key")
                opt["projection_weight"] = c["projection_weight"]
            if c.get("altered_weight", 0):
                opt["altered_weight"] = c["altered_weight"]
            if c.get("class_adjust") is not None:
                opt["extra"] = c["class_adjust"]
            forecasts.append(simulate(c["t"],c["cfg"],**opt))
        estimated=torch.stack(forecasts)
        anchored=actual[best]+estimated-estimated[:,best:best+1]
        historical=[]
        for number,name in receipts:
            if args.base_through and number<=args.base_through:
                continue
            eligible=torch.tensor([known[n][2]<number for n in names])
            previous=int(actual.masked_fill(~eligible,-torch.inf).argmax())
            i=names.index(name)
            frozen=actual[previous]+estimated[:,i]-estimated[:,previous]
            historical.append({"id":number,"actual":float(actual[i]),
                "median":float(frozen.quantile(.5)),"errors":(frozen-actual[i]).tolist()})
        result={"frozen_at":args.base_through,"latest":receipts[-1][0],
                "parent":names[best],"official":float(actual[best]),"historical":historical,"models":{}}
        for n in args.forecast:
            if n not in all_names:
                continue
            z=anchored[:,all_names.index(n)]
            result["models"][n]={"median":float(z.quantile(.5)),"range":[float(z.min()),float(z.max())],
                                 "ahead":int((z>actual[best]).sum()),"configs":len(z)}
        if args.forecast_output:
            path=args.forecast_output
            temp=path.with_suffix(".partial");temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(path)
        print(json.dumps(result),flush=True)
        return
    def score_derivatives(t,cfg):
        derivatives=[]
        for j in range(len(t)):
            step=torch.zeros_like(t);step[j]=.12*scale[j]
            derivatives.append((simulate(t+step,cfg)-simulate(t-step,cfg))/.24)
        return torch.stack(derivatives,1)

    def update(t,cfg,weights,start,jac):
        weighted=jac*weights[:,None]
        normal=jac.T@weighted+.025**2*torch.eye(len(t),dtype=torch.float64)
        proposal=torch.linalg.solve(normal,weighted.T@(actual-start)).clamp(-1,1)
        new=t+.65*proposal.float()*scale
        return new,simulate(new,cfg)
    def fit_classes(t,cfg):
        zero=t.new_zeros(3)
        baseline=simulate(t,cfg,zero)
        steps=torch.diag(torch.tensor([.13,.13,.10],dtype=t.dtype))
        jac=torch.stack([(simulate(t,cfg,h)-simulate(t,cfg,-h))/(2*h.abs().max())
                         for h in steps],dim=1)
        def estimate(weights):
            weighted=jac*weights[:,None]
            gram=jac.T@weighted+.045**2*torch.eye(3,dtype=torch.float64)
            return torch.linalg.solve(gram,weighted.T@(actual-baseline)).float().clamp(-.75,.75)
        checks=[]
        observed_ids=torch.tensor([known[n][2] for n in names])
        for number,name in receipts:
            if args.base_through and number<=args.base_through:
                continue
            # Evaluate the change versus the best previously scored model. The
            # held-out receipt and all later scores are excluded from this fit.
            prior_mask=observed_ids<number
            weights=base_weights*prior_mask
            anchor=int(actual.masked_fill(~prior_mask,-torch.inf).argmax())
            i=names.index(name)
            pred=simulate(t,cfg,estimate(weights))
            plain=float(actual[anchor]+baseline[i]-baseline[anchor])
            adjusted=float(actual[anchor]+pred[i]-pred[anchor])
            checks.append({"id":number,"baseline":plain,"adjusted":adjusted,"actual":float(actual[i])})
        if not checks:
            raise ValueError("No later receipt available to validate the frozen fit")
        plain=sum(abs(x["baseline"]-x["actual"]) for x in checks)/len(checks)
        adjusted=sum(abs(x["adjusted"]-x["actual"]) for x in checks)/len(checks)
        coefficient=estimate(base_weights) if adjusted<plain else zero
        return coefficient.tolist(),{"config":list(cfg),"used":adjusted<plain,
              "temporal_mae_before":plain,"temporal_mae_after":adjusted,"checks":checks}

    def fit_temporal_grid(t, cfg, parameter):
        grid = [0., .10, .25, .5, .75, 1.] if parameter == "shadow_weight" else [0., .25, .5, .75, 1.]
        predicted = torch.stack([simulate(t, cfg, **{parameter: w}) for w in grid])
        observed_ids = torch.tensor([known[n][2] for n in names])
        runs = []
        for number, name in receipts:
            previous = observed_ids < number
            anchor = int(actual.masked_fill(~previous, -torch.inf).argmax())
            i = names.index(name)
            forecast = actual[anchor] + predicted[:, i] - predicted[:, anchor]
            runs.append((number, forecast, actual[i]))

        def choose(subset):
            errors = torch.stack([(forecast - score).abs() for _, forecast, score in subset])
            # Prefer smaller changes when historical differences barely distinguish them.
            loss = errors.mean(0) + .0005 * torch.tensor(grid)
            return int(loss.argmin())

        checks = []
        for i, (number, forecasts, score) in enumerate(runs):
            if number <= args.base_through or i == 0:
                continue
            k = choose(runs[:i])
            checks.append({"id": number, "weight": grid[k], "actual": float(score),
                           "baseline": float(forecasts[0]), "adjusted": float(forecasts[k])})
        if not checks:
            raise ValueError("A newer official result is needed to validate the projection")
        before = [abs(x["baseline"] - x["actual"]) for x in checks]
        after = [abs(x["adjusted"] - x["actual"]) for x in checks]
        wins = sum(a < b for a, b in zip(after, before))
        used = sum(after) < sum(before) and 2 * wins >= len(checks)
        result = grid[choose(runs)] if used else 0.
        return result, {"config": list(cfg), "used": bool(result), "weight": result,
                        "temporal_mae_before": sum(before) / len(before),
                        "temporal_mae_after": sum(after) / len(after), "checks": checks}

    def fit_response(t, cfg):
        dims = len(response_basis["clean"])
        zero = torch.zeros(dims)
        baseline = simulate(t,cfg)
        steps = .15 * torch.eye(dims)
        jac = torch.stack([(simulate(t,cfg,response=h) - simulate(t,cfg,response=-h))/.30
                           for h in steps], 1)
        observed = torch.tensor([known[n][2] for n in names])
        def estimate(prior, anchor):
            # Differences in KL between submitted distributions cancel the gold entropy.
            # Anchor the score residual to a model that had already been evaluated.
            x = jac - jac[anchor]
            residual = (actual-baseline) - (actual-baseline)[anchor]
            wt = base_weights * prior * (actual > 0)
            weighted = x * wt[:,None]
            gram = x.T @ weighted + .035**2*torch.eye(dims,dtype=torch.float64)
            coefficients = torch.linalg.solve(gram, weighted.T @ residual).float()
            return coefficients / (coefficients.norm()/.80).clamp_min(1)
        checks = []
        for number, name in receipts:
            if number <= args.base_through:
                continue
            prior = observed < number
            anchor = int(actual.masked_fill(~prior,-torch.inf).argmax())
            i = names.index(name)
            predicted = simulate(t,cfg,response=estimate(prior,anchor))
            checks.append({"id":number,"actual":float(actual[i]),
                           "baseline":float(actual[anchor]+baseline[i]-baseline[anchor]),
                           "adjusted":float(actual[anchor]+predicted[i]-predicted[anchor])})
        if not checks:
            raise ValueError("A later public receipt is needed for the response comparison")
        before = [abs(x["baseline"]-x["actual"]) for x in checks]
        after = [abs(x["adjusted"]-x["actual"]) for x in checks]
        use = sum(after)<sum(before) and 2*sum(a<b for a,b in zip(after,before))>=len(checks)
        coefficients = estimate(torch.ones_like(observed).bool(), int(actual.argmax())) if use else zero
        return coefficients.tolist(), {"config": list(cfg), "used": use, "coefficients": coefficients.tolist(),
               "temporal_mae_before":sum(before)/len(checks),"temporal_mae_after":sum(after)/len(checks),"checks":checks}

    report=[];adapted=[]
    for c in configs:
        t,cfg=c["t"].float(),c["cfg"]
        if args.response_cache:
            coefficients, audit = fit_response(t, cfg)
            adapted.append(dict(c, response_coefficients=coefficients))
            report.append(audit)
            continue
        if temporal:
            parameter = "shadow_weight" if args.shadow_cache else "projection_weight" if args.projection_cache else "altered_weight"
            coefficient, audit = fit_temporal_grid(t, cfg, parameter)
            adapted.append(dict(c, **{parameter: coefficient}))
            report.append(audit)
            continue
        if args.class_features:
            extra,audit=fit_classes(t,cfg)
            adapted.append(dict(c,class_adjust=extra))
            report.append(audit)
            continue
        before=simulate(t,cfg);independent=base_weights.clone();independent[last]=0
        jac=score_derivatives(t,cfg)
        _,held=update(t,cfg,independent,before,jac)
        after_t,after=update(t,cfg,base_weights,before,jac)
        accept=abs(held[last]-actual[last])<=abs(before[last]-actual[last])+.003
        keep=after_t if accept else t
        adapted.append(dict(c,t=keep.cpu()))
        report.append({"config":list(cfg),"used":bool(accept),"previous":float(before[last]),
            "held_out":float(held[last]),"official":float(actual[last]),
            "fit_mae_before":float((before-actual).abs().mean()),
            "fit_mae_after":float((after-actual).abs().mean())})
    last_id=receipts[-1][0]
    stem=f"codex_fit_classes_recu_{last_id}" if args.class_features else f"codex_fit_recu_{last_id}"
    if temporal:
        stem = (f"codex_fit_projection_{args.projection_key}_recu_{last_id}" if args.projection_cache
                else f"codex_fit_alteres_recu_{last_id}")
        if args.shadow_cache:
            stem = f"codex_fit_hors_pli_recu_{last_id}"
            if args.shadow_mode != "raw":
                stem = f"codex_fit_hors_pli_{args.shadow_mode}_recu_{last_id}"
        if args.response_cache:
            stem = f"codex_fit_{args.response_label}_recu_{last_id}"
            if args.transfer_fit:
                by_cfg = {tuple(c["cfg"]): c["response_coefficients"] for c in adapted}
                empty = [0.] * len(response_basis["clean"])
                adapted = [dict(c,response_coefficients=by_cfg.get(tuple(c["cfg"]),empty))
                           for c in recipient_configs]
        elif args.transfer_fit:
            adapted = transfer_weights(adapted, recipient_configs, parameter)
        if args.projection_cache:
            for fit in adapted:
                fit["projection_key"] = args.projection_key
        if args.shadow_cache:
            for fit in adapted:
                fit["shadow_mode"] = args.shadow_mode
    dest=root/"cache"/(stem+".pt")
    temp=dest.with_suffix(".partial");torch.save(adapted,temp);temp.replace(dest)
    data={"latest":last_id,"models":len(names),"observed_scores":dict(zip(names,actual.tolist())),
          "sample":len(sample),"base_through":args.base_through,"checks":report}
    if args.transfer_fit:
        data["transferred_to"] = str(args.transfer_fit)
        if args.response_cache:
            data["applied_weights"] = [{"cfg": c["cfg"], "coefficients": c["response_coefficients"]} for c in adapted]
        else:
            data["applied_weights"] = [{"cfg": c["cfg"], "weight": c[parameter]} for c in adapted]
    path=root/"output"/(stem+".json");path.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data),flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    main()
