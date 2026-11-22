"""Build ensemble targets for training from the team's locally fitted score models."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def rank(difficulty):
    values = torch.empty_like(difficulty)
    values[difficulty.argsort()] = (torch.arange(len(values),dtype=values.dtype)+.5)/len(values)
    return values


def class_features(labels, crop_loss, augmentation_loss):
    counts=torch.bincount(labels,minlength=100).float()
    removed=counts[labels].clamp_min(1).log()
    removed=((removed-removed.mean())/removed.std().clamp_min(.1)).clamp(-2,2)
    stability=rank(crop_loss)-rank(augmentation_loss)
    stability=(stability/stability.std().clamp_min(.15)).clamp(-2,2)
    classes=counts.log1p()
    classes=((classes-classes.mean())/classes.std().clamp_min(.1)).clamp(-2,2)
    return removed,stability,classes


def adjust_classes(pi, logits, features, coefficients):
    removed,stability,classes=features
    coefficients=torch.as_tensor(coefficients,dtype=pi.dtype,device=pi.device)
    pi=(torch.logit(pi.clamp(1e-6,1-1e-6))+coefficients[0]*removed+
        coefficients[1]*stability).sigmoid()
    return pi,logits+coefficients[2]*classes


def projection_delta(path, key, labels):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.equal(cache["y"].long(), labels):
        raise ValueError("Projection and calibration use different image labels or order")
    delta = cache[key].float()
    if delta.shape != (len(labels), 100) or not torch.isfinite(delta).all():
        raise ValueError("Invalid per-image projection in " + str(path))
    return delta


def altered_probabilities(clean, labels, moments, weight):
    if not weight:
        return clean
    rows = torch.arange(len(labels), device=labels.device)
    true = clean[rows, labels].clamp(1e-7, 1-1e-7)
    alternatives = clean.clone()
    alternatives[rows, labels] = 0
    alternatives /= alternatives.sum(1, keepdim=True).clamp_min(1e-14)
    # Squaring a T=2 probability distribution gives its unnormalised T=1 probabilities.
    concentration = alternatives.square().sum(1).clamp_min(1e-20).log()
    ce = F.softplus(concentration - 2*torch.logit(true))
    ratio = moments[3] / moments[1].clamp_min(1e-5)
    shift = moments[2] - moments[0] + (ratio - 1) * (ce - ce.mean())
    requested_ce = (ce + weight*shift).clamp(.0001, 32.)
    odds = .5 * (concentration - requested_ce.expm1().log())
    new_true = odds.sigmoid()
    result = (1-new_true[:, None]) * alternatives
    result[rows, labels] = new_true
    return result


def shadow_targets(path, labels):
    pack = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.equal(pack["y"].long(), labels) or len(pack["folds"].unique()) != 4:
        raise ValueError("Auxiliary probabilities must cover the same images in four source folds")
    result = []
    for view in ("clean", "altered"):
        q = pack[view].float()
        if q.shape != (len(labels), 100) or not torch.isfinite(q).all() or (q < 0).any():
            raise ValueError("Invalid auxiliary probabilities for " + view)
        torch.testing.assert_close(q.sum(1), torch.ones(len(q)), atol=2e-5, rtol=1e-5)
        result.append(q)
    return result


def shadow_distribution(base, auxiliary, labels, mode):
    if mode == "raw":
        return auxiliary
    rows = torch.arange(len(labels), device=labels.device)
    true = base[rows, labels]
    ranking = auxiliary[rows, labels].argsort(stable=True)
    transferred = torch.empty_like(true)
    transferred[ranking] = true.sort().values
    alternative = base.clone()
    alternative[rows, labels] = 0
    alternative /= alternative.sum(1, keepdim=True).clamp_min(1e-12)
    result = (1-transferred[:, None]) * alternative
    result[rows, labels] = transferred
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-file", type=Path, default=Path("cache/lina_feat/_fit.pt"))
    parser.add_argument("--output", type=Path, default=Path("cache/codex_gold_public.pt"))
    parser.add_argument("--projection-cache", type=Path)
    parser.add_argument("--shadow-cache", type=Path)
    parser.add_argument("--response-cache", type=Path)
    args = parser.parse_args()
    base=Path("cache/lina_feat")
    def load(name):
        return torch.load(base/name,map_location="cpu",weights_only=True)
    meta,orig=load("_meta.pt"),load("_orig.pt")
    configs=torch.load(args.fit_file,map_location="cpu",weights_only=True)
    y=meta["y"].long()
    nm=orig["nonmem"].float()
    geo=[0,1,2,3,10]
    ce=torch.stack([F.cross_entropy(nm[:,i],y,reduction="none") for i in range(nm.shape[1])],1)
    features=class_features(y,ce[:,geo].mean(1),meta["diff"].float())
    difficulties={"aug":rank(meta["diff"].float()),"nmg":rank(ce[:,geo].mean(1)),
                  "nmall":rank(ce.mean(1))}
    bases={"za":meta["za"].float(),"nmg":nm[:,geo].mean(1),"clean":orig["clean"].float(),
           "p2za":2*meta["p2"].float().clamp_min(1e-9).log()}
    mask=F.one_hot(y,100).bool()
    projections = {}
    for config in configs:
        if config.get("projection_weight", 0):
            if args.projection_cache is None:
                parser.error("The calibrated fit requires --projection-cache")
            key = config["projection_key"]
            if key not in projections:
                projections[key] = projection_delta(args.projection_cache, key, y)
    if any(c.get("shadow_weight", 0) for c in configs) and not args.shadow_cache:
        parser.error("The calibrated fit requires --shadow-cache")
    shadow = shadow_targets(args.shadow_cache, y) if args.shadow_cache else None
    uses_response = any(any(c.get("response_coefficients", [])) for c in configs)
    if uses_response:
        if not args.response_cache:
            parser.error("The calibrated fit requires --response-cache")
        from directions_gold import tilt
        response = torch.load(args.response_cache, map_location="cpu", weights_only=True)
        if not torch.equal(response["y"], y):
            raise ValueError("Model responses and targets use different image order")
    distributions, alternate_distributions, moments, settings = [], [], [], []
    for model in configs:
        family,source,difficulty=model["cfg"]
        t=model["t"].float()
        ranks=difficulties[difficulty]
        pi=torch.sigmoid(t[5])+(torch.sigmoid(t[6])-torch.sigmoid(t[5]))*torch.sigmoid(
            (torch.sigmoid(t[7])-ranks)/.04)
        base_logits = bases[source]
        weight = model.get("projection_weight", 0)
        if weight:
            base_logits = base_logits + weight * projections[model["projection_key"]]
        logits=base_logits/(2*torch.exp(t[4]))
        if "class_adjust" in model:
            pi,logits=adjust_classes(pi,logits,features,model["class_adjust"])
        alt=logits.masked_fill(mask,-torch.inf).softmax(1)
        exact = pi[:,None]*mask.float()+(1-pi[:,None])*alt
        m=F.softplus(t[:4]);moments.append(m)
        altered = altered_probabilities(exact, y, m, model.get("altered_weight", 0))
        shadow_weight = model.get("shadow_weight", 0)
        if shadow_weight:
            mode = model.get("shadow_mode", "raw")
            exact = torch.lerp(exact, shadow_distribution(exact,shadow[0],y,mode), shadow_weight)
            altered = torch.lerp(altered, shadow_distribution(altered,shadow[1],y,mode), shadow_weight)
        coefficients = model.get("response_coefficients", [])
        if any(coefficients):
            direction = torch.tensor(coefficients)
            exact = tilt(exact, response["clean"].float(), direction)
            altered = tilt(altered, response[family].float(), direction)
        distributions.append(exact)
        alternate_distributions.append(altered)
        settings.append({"family":family,"source":source,"difficulty":difficulty,
                         "moments":m.tolist(),"mean_true_probability":float(pi.mean()),
                         "loo_log_error":float(model["loo_mae_log"]),
                         "class_adjust":model.get("class_adjust"),
                         "projection_weight":weight, "altered_weight":model.get("altered_weight",0),
                         "shadow_weight":shadow_weight, "response_coefficients":coefficients})
    q=torch.stack(distributions).mean(0)
    assert torch.isfinite(q).all() and torch.allclose(q.sum(1),torch.ones(len(q)),atol=2e-5)
    m=torch.stack(moments)
    result={"y":y,"probability":q,"members":torch.stack(distributions),
            "moments":m.median(0).values,"individual_moments":m,
            "configurations":settings}
    if uses_response or any(c.get("altered_weight", 0) or c.get("shadow_weight", 0) for c in configs):
        result["altered_members"] = torch.stack(alternate_distributions)
    dest=args.output;temp=dest.with_suffix(".partial")
    torch.save(result,temp);temp.replace(dest)
    info={"images":len(y),"members":len(configs),"moments":result["moments"].tolist(),
          "settings":settings,"mean_true_probability":float(q[torch.arange(len(y)),y].mean())}
    info["fit_file"]=str(args.fit_file)
    (Path("output")/(dest.stem+".json")).write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info),flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    main()
