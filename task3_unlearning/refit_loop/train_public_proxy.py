"""Distill calibrated probability targets while fitting the separate gold loss moments."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F

from calibrate_losses import split_images
from refine_loss_moments import moments, predict
from three_term_unlearning import crop_reference, fixed_views, retained_crops


def kl(q,z):
    return F.kl_div((z/2).log_softmax(1),q,reduction="batchmean")


def control_loss(teacher, student, decision_temperature):
    distance = kl((teacher / 2).softmax(1), student)
    if decision_temperature is None:
        return distance
    posterior = teacher.softmax(1)
    reference = (posterior * (teacher / decision_temperature).softmax(1)).sum(1).mean()
    agreement = (posterior * (student / decision_temperature).softmax(1)).sum(1).mean()
    deficit = F.relu(reference.clamp_min(1e-8).log() - agreement.clamp_min(1e-8).log())
    # Utility multiplies decision agreement by 1 / (1 + KL).
    # Keep the existing KL gradient scale while including the agreement gradient.
    return distance + (1 + distance.detach()) * deficit


def frozen_tokens(model, normalized_images, train_blocks):
    x = model.patch_embed(normalized_images)
    x = model.norm_pre(model.patch_drop(model._pos_embed(x)))
    return model.blocks[:-train_blocks](x)


def predict_from_tokens(model, tokens, train_blocks):
    x = model.blocks[-train_blocks:](tokens)
    return model.forward_head(model.norm(x)).float()


def student_targets(path, data, device):
    targets = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.equal(targets["y"], data["y"]) or not torch.equal(
        targets["source_moments"], data["individual_moments"]
    ):
        raise ValueError("Student targets and evaluation ensemble are misaligned")
    shape = (2, *data["members"].shape[1:])
    for split in ("fit", "full"):
        probability = targets[split]
        if probability.shape != shape or not torch.isfinite(probability).all() or (probability < 0).any():
            raise ValueError(f"Invalid student target probabilities for {split}")
        sums = probability.sum(-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=2e-5, rtol=0):
            raise ValueError(f"Student target probabilities do not sum to one for {split}")
        targets[split] = probability.to(device)
    return targets


def fitted_strengths(configurations, device):
    result=[]
    for config in configurations:
        name=config["family"]
        if not name.startswith("aug") or not name[3:].isdigit():
            raise ValueError(f"The fitted augmentation is unavailable: {name}")
        strength=int(name[3:])/100
        if not 0<=strength<=1:
            raise ValueError(f"Augmentation strength outside the supported interval: {strength}")
        result.append(strength)
    return torch.tensor(result,device=device)


def augmentation_strengths(draw, strengths):
    if strengths is None:
        return .17 + .53 * draw
    return strengths[(draw * len(strengths)).long()]


@torch.no_grad()
def source_loss_population(recipe, source, images, labels, indices, strengths, stratify_loss=False):
    generator = torch.Generator(device=labels.device).manual_seed(51483)
    raw = torch.zeros((2, 2), device=labels.device)
    profiles = []
    for ids in indices.split(32):
        x = recipe.to01(images[ids])
        draws = torch.rand(len(ids), device=labels.device, generator=generator)
        altered = recipe.augment(x, augmentation_strengths(draws, strengths), generator)
        logits = predict(recipe, source, torch.cat((x, altered)))
        losses = F.cross_entropy(logits, labels[ids].repeat(2), reduction="none").reshape(2, -1)
        raw[:, 0] += losses.sum(1)
        raw[:, 1] += losses.square().sum(1)
        if stratify_loss:
            profiles.append(losses.mean(0))
    return raw / len(indices), torch.cat(profiles) if profiles else None


def training_batches(indices, difficulty, generator):
    if difficulty is None:
        return indices[torch.randperm(len(indices), generator=generator, device=indices.device)].split(32)
    count = math.ceil(len(indices) / 32)
    ordered = indices[difficulty.argsort()]
    # Distribute each small interval of source loss across different batches.
    assigned = torch.cat([torch.randperm(count, generator=generator, device=indices.device)[:len(group)]
                          for group in ordered.split(count)])
    sizes = assigned.bincount(minlength=count).tolist()
    shuffle = torch.randperm(len(ordered), generator=generator, device=indices.device)
    by_batch = shuffle[assigned[shuffle].argsort(stable=True)]
    return ordered[by_batch].split(sizes)


def paired_population_moments(losses, source, raw):
    # Both networks see identical images. Their paired difference estimates the
    # change in the full population's first and second raw moments.
    change = losses - source
    mean = raw[0] + change.mean()
    second = raw[1] + (change * (losses + source)).mean()
    return torch.stack((mean, (second - mean.square()).clamp_min(1e-8).sqrt()))


def receipt_training_seed(args):
    default = 28499
    if args.receipt is None:
        return default
    settings = {"source": args.source.name, "fraction": args.fraction,
                "retention_weight": args.retention_weight, "anchor_strategy": args.anchor_strategy,
                "reference_gold": str(args.reference_gold) if args.reference_gold else None}
    for path in args.output.parent.glob("codex_prob_recu_*.json"):
        match = re.fullmatch(r"codex_prob_recu_(\d+)(?:_reprise)?", path.stem)
        if match and int(match[1]) < args.receipt:
            previous = json.loads(path.read_text())
            if (previous.get("epochs") and bool(previous.get("population_moments")) == args.population_moments
                    and bool(previous.get("stratify_loss")) == args.stratify_loss
                    and previous.get("decision_temperature") == args.decision_temperature
                    and previous.get("epoch_limit", 4) == args.epochs
                    and all(previous.get(k) == v for k, v in settings.items())):
                return default + args.receipt
    return default


def balance_gradients(forget, retain):
    dot = torch.stack([torch.dot(f.reshape(-1), r.reshape(-1)) for f, r in zip(forget, retain)]).sum()
    norm_f = torch.stack([torch.dot(g.reshape(-1), g.reshape(-1)) for g in forget]).sum()
    norm_r = torch.stack([torch.dot(g.reshape(-1), g.reshape(-1)) for g in retain]).sum()
    opposed = torch.clamp_max(dot, 0.)
    # Remove only the component that would increase the other objective to first order.
    forget_factor = 1 - opposed / norm_f.clamp_min(1e-20)
    retain_factor = 1 - opposed / norm_r.clamp_min(1e-20)
    gradient = [forget_factor * f + retain_factor * r for f, r in zip(forget, retain)]
    cosine = dot / (norm_f.sqrt() * norm_r.sqrt()).clamp_min(1e-20)
    return gradient, cosine


def conditional_view_target(probability, labels, parent_clean, parent_view, original_clean, original_view):
    # Transfer the observed change in alternative classes after removing the pseudo-label.
    mask=F.one_hot(labels,num_classes=probability.shape[1]).bool()
    def alternatives(z):
        return (z/2).masked_fill(mask,-1e4).log_softmax(1)
    delta=(alternatives(original_view)-alternatives(original_clean)).clamp(-2,2)
    wrong=(probability.clamp_min(1e-12).log()+.5*delta).masked_fill(mask,-1e4).softmax(1)
    row=torch.arange(len(labels),device=labels.device)
    def odds(p):
        return torch.logit(p[row,labels].clamp(1e-6,1-1e-6))
    keep=(odds(probability)+.35*(odds(parent_view)-odds(parent_clean)).clamp(-2,2)).sigmoid()
    return (1-keep[:,None])*wrong+keep[:,None]*mask


@torch.no_grad()
def control_weights(recipe, original, source, pixels, *, logits_batches=None):
    if pixels is None:
        return None, {}
    drifts, labels = [], []
    for batch in pixels.split(48):
        x = batch.float()/255
        reference = predict(recipe,original,x)
        if logits_batches is not None:
            logits_batches.append(reference)
        old = (reference/2).log_softmax(1)
        current = (predict(recipe,source,x)/2).log_softmax(1)
        drifts.append((old.exp()*(old-current)).sum(1).clamp_min(0.))
        labels.append(reference.argmax(1))
    drift, labels = torch.cat(drifts),torch.cat(labels)
    # Balance class coverage and give extra weight to reliable crops where behavior changed.
    ratio = drift.clamp(max=drift.quantile(.95))/drift.median().clamp_min(1e-5)
    per_class = torch.bincount(labels,minlength=100).float()[labels]
    weights = (.45+.55*ratio.sqrt())/per_class.sqrt()
    weights = weights/weights.sum()
    return weights, {"effective_samples":float(1/weights.square().sum()),
                     "uniform_drift":float(drift.mean()),"sampled_drift":float(weights@drift)}


@torch.no_grad()
def wide_control_crops(recipe, original, images, labels, indices, seed, repeats, cap):
    generator=torch.Generator(device=indices.device).manual_seed(seed)
    pixels,classes,confidence,scales=[],[],[],[]
    for _ in range(repeats):
        for ids in indices.split(32):
            x=recipe.to01(images[ids])
            def random():
                return torch.rand(len(ids),device=x.device,generator=generator)
            scale=.57+.32*random()
            flip=torch.ones_like(scale)
            crop=recipe.affine(x,scale,8*(random()-.5),flip,2*random()-1,2*random()-1)
            packed=(crop.clamp(0,1)*255).round().to(torch.uint8)
            crop=packed.float()/255
            mild=recipe.augment(crop,.15,generator)
            original_views=predict(recipe,original,torch.cat((crop,crop.flip(-1),mild))).chunk(3)
            probs,guesses=zip(*(z.softmax(1).max(1) for z in original_views))
            stable=(guesses[0]==guesses[1])&(guesses[0]==guesses[2])&(guesses[0]!=labels[ids])
            reliability=torch.stack(probs).amin(0)
            stable &= reliability>=.40
            if stable.any():
                pixels.append(packed[stable])
                classes.append(guesses[0][stable])
                confidence.append(reliability[stable])
                scales.append(scale[stable])
    if not pixels:
        return None,{"count":0}
    pixels,classes,confidence,scales=map(torch.cat,(pixels,classes,confidence,scales))
    selected=[]
    for label in classes.unique():
        group=(classes==label).nonzero().flatten()
        selected.append(group[confidence[group].argsort(descending=True)[:cap]])
    selected=torch.cat(selected)
    return pixels[selected],{"count":len(selected),"classes":len(classes[selected].unique()),
            "median_side_fraction":float(scales[selected].median()),
            "median_confidence":float(confidence[selected].median())}


@torch.no_grad()
def measure_control_set(recipe, model, controls):
    if not controls:
        return {}
    new=torch.cat([predict(recipe,model,x.float()/255) for x,_ in controls])
    orig=torch.cat([z for _,z in controls])
    return {"agreement":float((orig.argmax(1)==new.argmax(1)).float().mean()),
            "kl":float(kl((orig/2).softmax(1),new))}


@torch.no_grad()
def assess(R, student, batches, controls, labels, probabilities, gold_stats, altered_probabilities=None):
    zlist=[[] for _ in range(5)]; olist=[[] for _ in range(5)]; indices=[]
    for ids, pixels, original in batches:
        indices.append(ids)
        choices=predict(R,student,pixels.float()).split(len(ids))
        for i,(z,o) in enumerate(zip(choices,original)):
            zlist[i].append(z);olist[i].append(o)
    ids=torch.cat(indices); zlist=list(map(torch.cat,zlist));olist=list(map(torch.cat,olist))
    if controls:
        zlist.append(torch.cat([predict(R,student,x.float()/255) for x,_ in controls]))
        olist.append(torch.cat([o for _,o in controls]))
    retention_metrics=[]
    for z,o in zip(zlist[3:],olist[3:]):
        retention_metrics.append(torch.stack(((z.argmax(1)==o.argmax(1)).float().mean(),
                                             kl((o/2).softmax(1),z))))
    agree,drift=torch.stack(retention_metrics).mean(0)
    retain=agree/(1+drift)
    scenarios=[]
    alternatives = probabilities if altered_probabilities is None else altered_probabilities
    for q,other,m in zip(probabilities,alternatives,gold_stats):
        loss=[];prediction=[]
        for v,inds in enumerate(((0,),(1,2))):
            z=torch.cat([zlist[i] for i in inds]);o=torch.cat([olist[i] for i in inds])
            y=labels[ids].repeat(len(inds))
            target=(q if v==0 else other)[ids].repeat(len(inds),1)
            goal=m.reshape(2,2)[v]
            def error(a):
                return (moments(F.cross_entropy(a,y,reduction="none"))-goal).abs().sum()
            loss.append((1-error(z)/error(o).clamp_min(1e-6)).clamp(0,1))
            prediction.append((1-kl(target,z)/kl(target,o).clamp_min(1e-6)).clamp(0,1))
        l=torch.stack(loss).mean();p=torch.stack(prediction).mean()
        scenarios.append([float(l),float(p),float(l*p*retain)])
    scores=torch.tensor([s[2] for s in scenarios])
    return {"quality":float(scores.quantile(.25)),"scenarios":scenarios,
            "retention":float(retain),"control_kl":float(drift),"agreement":float(agree)}


def train(R, path, source, original, images, labels, ids_all, bank, q, target_moments,
          passes, fraction, callback, retention_weight, prediction_weight, weights=None,
          altered_moment_weight=1.,view_teacher="parent",gradient_stats=None,altered_q=None,train_blocks=2,
          strengths=None, population_moments=False, training_seed=28499, stratify_loss=False,
          decision_temperature=None, share_frozen_blocks=False, bank_logits=None):
    model=R.load_model(path)
    if not 1 <= train_blocks <= len(model.blocks):
        raise ValueError("Requested block count outside the model")
    params=list(R.select_params(model,f"last{train_blocks}"))
    if share_frozen_blocks and (model.training or source.training):
        raise ValueError("Shared frozen blocks require deterministic evaluation mode")
    if bank_logits is not None and (bank is None or weights is None or original.training or view_teacher != "parent"):
        bank_logits = None
    groups = params
    if train_blocks > 2:
        extra = [v for block in model.blocks[-train_blocks:-2] for v in block.parameters() if v.requires_grad]
        earlier = {id(v) for v in extra}
        groups = [{"params": [v for v in params if id(v) not in earlier]},
                  {"params": extra, "lr": 8e-6}]
    opt=torch.optim.AdamW(groups,lr=2e-5,weight_decay=0.)
    scaler=torch.amp.GradScaler("cuda")
    g=torch.Generator(device=labels.device).manual_seed(training_seed)
    per_epoch=math.ceil(len(ids_all)/32)
    updates=math.ceil(per_epoch*passes)
    checkpoints=(.25,.5,*range(1,math.ceil(passes)+1),passes)
    marks={max(1,round(per_epoch*x)) for x in checkpoints}
    gradient_cosines = []
    raw_population = None
    difficulty = None
    if population_moments:
        raw_population, difficulty = source_loss_population(R, source, images, labels, ids_all,
                                                           strengths, stratify_loss)
        mean, second = raw_population.unbind(1)
        spread = (second - mean.square()).clamp_min(1e-8).sqrt()
        population_targets = torch.lerp(torch.stack((mean, spread), dim=1),
                                        target_moments.reshape(2, 2), fraction)
    step=0
    for _ in range(math.ceil(passes)):
        for ids in training_batches(ids_all, difficulty, g):
            x=R.to01(images[ids]);n=len(x);y=labels[ids]
            if n<2:continue
            with torch.no_grad():
                draw=torch.rand(n,device=y.device,generator=g)
                strength=augmentation_strengths(draw,strengths)
                altered=R.augment(x,strength,g);forget=torch.cat((x,altered))
                ai=ids_all[torch.randint(len(ids_all),(32,),device=y.device,generator=g)]
                anchors=R.make_anchor(R.to01(images[ai]),g,["crop","mos2","mix","cutmix"])
                if bank is not None:
                    if weights is None:
                        bi=torch.randint(len(bank),(16,),device=y.device,generator=g)
                    else:
                        bi=torch.multinomial(weights,16,replacement=True,generator=g)
                    anchors=torch.cat((anchors,bank[bi].float()/255))
                inputs=torch.cat((forget,anchors))
                if share_frozen_blocks:
                    with torch.autocast("cuda",dtype=torch.float16):
                        tokens=frozen_tokens(model,R.norm(inputs),train_blocks)
                        src=predict_from_tokens(source,tokens[:len(forget)],train_blocks)
                    if step==0:
                        direct=predict(R,source,forget)
                        diff=(src-direct).abs()
                        if diff.mean()>.008 or diff.max()>.14:
                            print("Independent source forward used",float(diff.mean()),float(diff.max()),flush=True)
                            share_frozen_blocks=False
                            src=direct
                else:
                    src=predict(R,source,forget)
                pc,pa=(src/2).softmax(1).chunk(2)
                clean=q[ids]
                altered_base=clean if altered_q is None else altered_q[ids]
                if view_teacher=="conditional":
                    ref=predict(R,original,torch.cat((forget,anchors)))
                    rc,ra,base=ref.split((n,n,len(anchors)))
                    corrected=conditional_view_target(altered_base,y,pc,pa,rc,ra)
                else:
                    if bank_logits is None:
                        base=predict(R,original,anchors)
                    else:
                        live=predict(R,original,anchors[:-len(bi)])
                        base=torch.cat((live,bank_logits[bi]))
                        if step==0:
                            direct=predict(R,original,anchors)
                            if not torch.allclose(base,direct,rtol=1e-5,atol=1e-6):
                                print("Direct forward used for control anchors",flush=True)
                                bank_logits=None
                            base=direct
                    factor=(pa.clamp_min(1e-9).log()-pc.clamp_min(1e-9).log()).clamp(-2,2)
                    corrected=(altered_base.clamp_min(1e-12).log()+.35*factor).softmax(1)
                gold=torch.lerp(torch.cat((pc,pa)),torch.cat((clean,corrected)),fraction)
                parent_loss=F.cross_entropy(src,y.repeat(2),reduction="none")
            if share_frozen_blocks:
                with torch.autocast("cuda",dtype=torch.float16):
                    z=predict_from_tokens(model,tokens,train_blocks)
            else:
                z=predict(R,model,inputs)
            zf,za=z.split((2*n,len(anchors)))
            individual=F.cross_entropy(zf,y.repeat(2),reduction="none")
            ml=0
            for j,(ls,previous) in enumerate(zip(individual.chunk(2),parent_loss.chunk(2))):
                if raw_population is None:
                    observed = moments(ls)
                    target = torch.lerp(moments(previous),target_moments.reshape(2,2)[j],fraction)
                else:
                    observed = paired_population_moments(ls, previous, raw_population[j])
                    target = population_targets[j]
                weight=altered_moment_weight if j else 1.
                ml=ml+weight*F.smooth_l1_loss(observed,target,beta=.5)
            forget_loss=3*ml+prediction_weight*kl(gold,zf)+.045*F.smooth_l1_loss(individual,parent_loss)
            retain_loss=retention_weight*control_loss(base,za,decision_temperature)
            opt.zero_grad(set_to_none=True)
            if gradient_stats is None:
                scaler.scale(forget_loss+retain_loss).backward()
            else:
                first=torch.autograd.grad(scaler.scale(forget_loss),params,retain_graph=True)
                second=torch.autograd.grad(scaler.scale(retain_loss),params)
                combined,cosine=balance_gradients(first,second)
                for parameter,grad in zip(params,combined):
                    parameter.grad=grad
                gradient_cosines.append(cosine.detach())
                del first,second,combined
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update()
            step+=1
            if callback is not None and step in marks:callback(step/per_epoch,model)
            if step>=updates:break
        if step>=updates:break
    if gradient_stats is not None and gradient_cosines:
        values=torch.stack(gradient_cosines)
        valid=values[torch.isfinite(values)]
        gradient_stats["finite_steps"]=len(valid)
        if len(valid):
            gradient_stats["opposed_fraction"]=float((valid<0).float().mean())
            gradient_stats["mean_cosine"]=float(valid.mean())
    return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source",type=Path);ap.add_argument("output",type=Path)
    ap.add_argument("--fraction",type=float,default=.5);ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--gold-cache",type=Path,default=Path("cache/codex_gold_public.pt"))
    ap.add_argument("--student-targets",type=Path)
    ap.add_argument("--fitted-augmentations",action="store_true")
    ap.add_argument("--population-moments",action="store_true")
    ap.add_argument("--stratify-loss",action="store_true")
    ap.add_argument("--decision-temperature",type=float)
    ap.add_argument("--share-frozen-blocks",action="store_true")
    ap.add_argument("--receipt",type=int)
    ap.add_argument("--reference-gold",type=Path)
    ap.add_argument("--retention-weight",type=float,default=21.)
    ap.add_argument("--prediction-weight",type=float,default=14.)
    ap.add_argument("--anchor-strategy",choices=("uniform","drift"),default="uniform")
    ap.add_argument("--altered-moment-weight",type=float,default=1.)
    ap.add_argument("--view-teacher",choices=("parent","conditional"),default="parent")
    ap.add_argument("--wide-controls",action="store_true")
    ap.add_argument("--balance-gradients",action="store_true")
    ap.add_argument("--train-blocks",type=int,default=2)
    a=ap.parse_args();assert 0<a.fraction<=1 and min(a.retention_weight,a.prediction_weight,a.altered_moment_weight)>0
    if a.stratify_loss and not a.population_moments:
        ap.error("Loss stratification requires the source population measurements")
    if a.decision_temperature is not None and not (math.isfinite(a.decision_temperature) and a.decision_temperature>0):
        ap.error("Decision temperature must be positive and finite")
    training_seed=receipt_training_seed(a)
    root=Path.cwd();sys.path.insert(0,str(root));import unlearn as R
    torch.manual_seed(814)
    data=torch.load(a.gold_cache,map_location="cpu",weights_only=True)
    if a.reference_gold:
        reference=torch.load(a.reference_gold,map_location="cpu",weights_only=True)
        if not torch.equal(data["y"],reference["y"]):
            raise ValueError("The target ensembles use different image labels")
        if "altered_members" in data or "altered_members" in reference:
            data["altered_members"]=torch.cat((data.get("altered_members",data["members"]),
                                               reference.get("altered_members",reference["members"])))
        for key in ("members","individual_moments"):
            data[key]=torch.cat((data[key],reference[key]))
        if a.fitted_augmentations:
            data["configurations"]=list(data["configurations"])+list(reference["configurations"])
    y=data["y"].long().to(R.DEV);qs=data["members"].to(R.DEV);ms=data["individual_moments"].to(R.DEV)
    alt = data["altered_members"].to(R.DEV) if "altered_members" in data else None
    if alt is not None and alt.shape != qs.shape:
        raise ValueError("Mismatched exact and altered target distributions")
    held_alt = alt[1::2] if alt is not None else None
    strengths=fitted_strengths(data["configurations"],R.DEV) if a.fitted_augmentations else None
    if strengths is not None and len(strengths)!=len(qs):
        raise ValueError("One augmentation is required for each fitted reference")
    targets = student_targets(a.student_targets, data, R.DEV) if a.student_targets else None
    if targets is None:
        fit_q, full_q = qs[::2].mean(0), qs.mean(0)
        fit_alt = alt[::2].mean(0) if alt is not None else None
        full_alt = alt.mean(0) if alt is not None else None
    else:
        fit_q, fit_alt = targets["fit"]
        full_q, full_alt = targets["full"]
    assert len(qs)>=4 and len(y)==3500
    fit,valid=split_images(y);images=R.load_images(root.parent/"forget_images.npz")
    original=R.load_model(root.parent/"model.pt").requires_grad_(False)
    source=R.load_model(a.source).requires_grad_(False)
    fixed=fixed_views(R,original,images,valid,61820)
    bank,counts=retained_crops(R,original,images,y,fit,4125,3,40,True)
    standard_count=len(bank) if bank is not None else 0
    wide_fit,wide_val,wide_reference={},{},None
    if a.wide_controls:
        broad,wide_fit=wide_control_crops(R,original,images,y,fit,83406,5,40)
        broader_hold,wide_val=wide_control_crops(R,original,images,y,valid,90317,3,12)
        wide_reference=crop_reference(R,original,broader_hold)
        if broad is not None:
            bank=torch.cat((bank,broad)) if bank is not None else broad
    collect_logits = (a.anchor_strategy == "drift" and a.view_teacher == "parent"
                      and not a.wide_controls and not original.training)
    logits_batches = [] if collect_logits else None
    weights,importance=(control_weights(R,original,source,bank,logits_batches=logits_batches)
                        if a.anchor_strategy=="drift" else (None,{}))
    bank_logits = torch.cat(logits_batches) if logits_batches else None
    del logits_batches
    if a.wide_controls and bank is not None and 0<standard_count<len(bank):
        if weights is None:
            weights=torch.ones(len(bank),device=y.device)
        weights[:standard_count]*=.5/weights[:standard_count].sum()
        weights[standard_count:]*=.5/weights[standard_count:].sum()
    hold,counts_h=retained_crops(R,original,images,y,valid,9001,3,12,True)
    controls=crop_reference(R,original,hold)
    baseline=assess(R,source,fixed,controls,y,qs[1::2],ms[1::2],held_alt)
    report={"source":a.source.name,"fraction":a.fraction,"baseline":baseline,
            "gold_cache":str(a.gold_cache),"retention_weight":a.retention_weight,
            "reference_gold":str(a.reference_gold) if a.reference_gold else None,
            "student_targets":str(a.student_targets) if a.student_targets else None,
            "student_strength":targets["strength"] if targets is not None else None,
            "augmentation_strengths":strengths.tolist() if strengths is not None else None,
            "population_moments":a.population_moments,
            "stratify_loss":a.stratify_loss,
            "decision_temperature":a.decision_temperature,
            "share_frozen_blocks":a.share_frozen_blocks,
            "training_seed":training_seed,
            "epoch_limit":a.epochs,
            "prediction_weight":a.prediction_weight,
            "train_blocks":a.train_blocks,
            "separate_altered_targets":alt is not None,
            "anchor_strategy":a.anchor_strategy,"importance":importance,
            "altered_moment_weight":a.altered_moment_weight,
            "view_teacher":a.view_teacher,
            "wide_fit":wide_fit,"wide_validation":wide_val,
            "fit_controls":counts,"validation_controls":counts_h,"epochs":[]}
    best=[baseline["quality"],0.]
    def save():
        path=a.output.with_suffix(".json");tmp=path.with_suffix(".partial");tmp.write_text(json.dumps(report,indent=2)+"\n");tmp.replace(path)
    def inspect(epoch,model):
        v=assess(R,model,fixed,controls,y,qs[1::2],ms[1::2],held_alt);report["epochs"].append({"epoch":epoch,**v})
        if v["quality"]>best[0]:best[:]=[v["quality"],epoch]
        save();print("EPOCH",epoch,v["quality"],v["retention"],flush=True)
    print("SOURCE",baseline,flush=True);save()
    trial_stats = {} if a.balance_gradients else None
    trial=train(R,a.source,source,original,images,y,fit,bank,fit_q,ms[::2].median(0).values,
                a.epochs,a.fraction,inspect,a.retention_weight,a.prediction_weight,weights,a.altered_moment_weight,
                a.view_teacher,trial_stats,fit_alt,a.train_blocks,
                strengths[::2] if strengths is not None else None,a.population_moments,training_seed,a.stratify_loss,
                a.decision_temperature,a.share_frozen_blocks,bank_logits)
    if trial_stats is not None:
        report["trial_gradient_alignment"]=trial_stats
    del trial,fixed;torch.cuda.empty_cache()
    report["selected_epoch"]=best[1];save()
    if not best[1]:print("Source leads the held-out configurations",flush=True);return
    final_stats = {} if a.balance_gradients else None
    final=train(R,a.source,source,original,images,y,torch.arange(len(y),device=y.device),bank,
                full_q,ms.median(0).values,best[1],a.fraction,None,a.retention_weight,a.prediction_weight,
                weights,a.altered_moment_weight,a.view_teacher,final_stats,full_alt,
                a.train_blocks,strengths,a.population_moments,training_seed,a.stratify_loss,
                a.decision_temperature,a.share_frozen_blocks,bank_logits)
    if final_stats is not None:
        report["final_gradient_alignment"]=final_stats
    fresh=fixed_views(R,original,images,valid,87022)
    report["fresh_source"]=assess(R,source,fresh,controls,y,qs,ms,alt)
    report["fresh_final"]=assess(R,final,fresh,controls,y,qs,ms,alt)
    report["wide_source"]=measure_control_set(R,source,wide_reference)
    report["wide_final"]=measure_control_set(R,final,wide_reference)
    save()
    tmp=a.output.with_name(a.output.stem+".partial.pt");R.save_checked(final,str(tmp));os.replace(tmp,a.output)
    print("READY",a.output,report["fresh_final"]["quality"],flush=True)


if __name__=="__main__":
    main()
