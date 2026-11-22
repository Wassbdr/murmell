"""Compare and train unlearning checkpoints with matched original-model references."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from calibrate_losses import split_images
from refine_loss_moments import ensemble_target, full_target, moments, predict, train


def teacher_from_seeds(cache, seeds, labels, base="prob"):
    values = []
    for seed in seeds:
        raw = torch.load(cache / f"raw3_K16_s0.75_seed{seed}.pt",
                         map_location="cpu", weights_only=True)
        assert torch.equal(labels.cpu(), raw["y"]) and raw["p2"].shape == (len(labels), 100)
        values.append(raw["p2" if base == "prob" else "za"].float())
    mean = torch.stack(values).mean(0)
    if base == "logit":
        assert torch.isfinite(mean).all()
        return mean.to(labels.device)
    assert torch.isfinite(mean).all() and torch.allclose(mean.sum(1), torch.ones(len(mean)), atol=0.002)
    return 2 * mean.clamp_min(1e-12).log().to(labels.device)


@torch.no_grad()
def fixed_views(recipe, original, images, indices, seed):
    generator = torch.Generator(device=images.device).manual_seed(seed)
    batches = []
    for ids in indices.split(32):
        x = recipe.to01(images[ids])
        views = (x, recipe.augment(x, 0.3, generator), recipe.augment(x, 0.7, generator),
                 recipe.make_anchor(x, generator, ["crop"]), recipe.make_anchor(x, generator, ["mos2"]))
        # Both models later receive exactly these bytes, including on altered views.
        packed = torch.cat(views).half()
        base = predict(recipe, original, packed.float()).split(len(ids))
        batches.append((ids, packed, base))
    return batches


@torch.no_grad()
def retained_crops(recipe, original, images, labels, indices, seed, repeats, per_class,
                   all_classes=False):
    generator = torch.Generator(device=images.device).manual_seed(seed)
    present = torch.bincount(labels, minlength=100) > 0
    crops, classes, confidence = [], [], []
    for _ in range(repeats):
        for ids in indices.split(32):
            crop = recipe.make_anchor(recipe.to01(images[ids]), generator, ["crop"])
            pixels = crop.mul(255).round().to(torch.uint8)
            x = pixels.float() / 255
            z, flip = predict(recipe, original, torch.cat((x, x.flip(-1)))).chunk(2)
            prob, guess = z.softmax(1).max(1)
            other_prob, other_guess = flip.softmax(1).max(1)
            keep = (guess == other_guess) & (guess != labels[ids])
            crops.append(pixels[keep])
            classes.append(guess[keep])
            confidence.append(torch.minimum(prob, other_prob)[keep])
    crops, classes, confidence = map(torch.cat, (crops, classes, confidence))
    novel = ~present[classes]
    pool = torch.ones_like(novel) if all_classes or int(novel.sum()) < per_class else novel
    chosen = []
    for cls in classes[pool].unique():
        ids = ((classes == cls) & pool).nonzero().flatten()
        chosen.append(ids[confidence[ids].argsort(descending=True)[:per_class]])
    if not chosen:
        return None, {"count":0}
    chosen = torch.cat(chosen)
    stats = {"count":len(chosen),"classes":len(classes[chosen].unique()),
             "median_confidence":float(confidence[chosen].median()),
             "class_absent_from_forget_labels":float(novel[chosen].float().mean())}
    return crops[chosen], stats


def strengthen_crop_labels(recipe, original, pixels):
    if pixels is None:
        return None, {}
    output, old_conf, new_conf, stable = [], [], [], []
    for batch in pixels.split(32):
        clean = batch.float() / 255
        with torch.no_grad():
            first = predict(recipe, original, clean)
            confidence, labels = first.softmax(1).max(1)
        current = clean
        for _ in range(3):
            current = current.detach().requires_grad_(True)
            logits = predict(recipe, original, current)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            gradient, = torch.autograd.grad(loss, current)
            gradient = F.avg_pool2d(gradient, 3, stride=1, padding=1)
            current = (current.detach() - gradient.sign() / 255).clamp(0, 1)
        with torch.no_grad():
            final, flipped = predict(recipe, original, torch.cat((current, current.flip(-1)))).chunk(2)
            prob = final.softmax(1).gather(1, labels[:,None]).flatten()
            accept = (final.argmax(1) == labels) & (flipped.argmax(1) == labels) & (prob > confidence)
            selected = torch.where(accept[:,None,None,None], current, clean)
            output.append(selected.mul(255).round().to(torch.uint8))
            old_conf.append(confidence)
            new_conf.append(torch.where(accept, prob, confidence))
            stable.append(accept)
    return torch.cat(output), {"original_confidence":float(torch.cat(old_conf).mean()),
            "strengthened_confidence":float(torch.cat(new_conf).mean()),
            "accepted_fraction":float(torch.cat(stable).float().mean())}


@torch.no_grad()
def crop_reference(recipe, original, pixels):
    if pixels is None:
        return None
    return [(x, predict(recipe, original, x.float() / 255)) for x in pixels.split(48)]


@torch.no_grad()
def per_view_targets(recipe, source, loss_model, batches, labels, fraction, spread=0):
    parts = [[] for _ in range(3)]
    for ids, views, _ in batches:
        x = views[:3*len(ids)].float()
        y = labels[ids].repeat(3)
        a = predict(recipe,source,x)
        b = a if source is loss_model else predict(recipe,loss_model,x)
        ce = torch.lerp(F.cross_entropy(a,y,reduction="none"),
                        F.cross_entropy(b,y,reduction="none"),fraction)
        for group,z in zip(parts,ensemble_target(a,y,ce,spread).split(len(ids))):
            group.append(z)
    return [torch.cat(parts[0]),torch.cat(parts[1]+parts[2])]


@torch.no_grad()
def measure(recipe, model, batches, labels, hypotheses, crop_controls=None, scale=1.0, per_view=None):
    ids_all, original, student = [], [[] for _ in range(5)], [[] for _ in range(5)]
    for ids, views, orig in batches:
        submitted = (scale * predict(recipe, model, views.float())).split(len(ids))
        ids_all.append(ids)
        for i in range(5):
            student[i].append(submitted[i])
            original[i].append(orig[i])
    ids = torch.cat(ids_all)
    original = [torch.cat(group) for group in original]
    student = [torch.cat(group) for group in student]
    o_control, s_control = torch.cat(original[3:]), torch.cat(student[3:])
    control_kl = float(recipe.kl_T(o_control, s_control).mean())
    agreement = float((o_control.argmax(1) == s_control.argmax(1)).float().mean())
    natural = {}
    if crop_controls:
        o = torch.cat([z for _, z in crop_controls])
        s = scale * torch.cat([predict(recipe, model, x.float() / 255) for x, _ in crop_controls])
        natural = {"kl":float(recipe.kl_T(o, s).mean()),
                   "agreement":float((o.argmax(1) == s.argmax(1)).float().mean())}
        control_kl = (2 * control_kl + natural["kl"]) / 3
        agreement = (2 * agreement + natural["agreement"]) / 3
    retain_proxy = agreement / (1 + control_kl)
    measured = []
    groups = [(name,clean[ids],alt[ids].repeat(2,1)) for name,clean,alt in hypotheses]
    if per_view is not None:
        groups.append(("per_view_distillation",per_view[0],per_view[1]))
    for name, clean_target, alt_target in groups:
        components = []
        for view_ids, gold in [([0], clean_target), ([1, 2], alt_target)]:
            orig = torch.cat([original[i] for i in view_ids])
            sub = torch.cat([student[i] for i in view_ids])
            y = labels[ids].repeat(len(view_ids))
            losses = [moments(F.cross_entropy(z, y, reduction="none")) for z in (orig, sub, gold)]
            d_orig, d_sub = [(m - losses[2]).abs().sum() for m in losses[:2]]
            kl_orig = recipe.kl_T(gold, orig).mean()
            kl_sub = recipe.kl_T(gold, sub).mean()
            assert d_orig > 1e-8 and kl_orig > 1e-8
            components.append({
                "loss_match":float((1 - d_sub / d_orig).clamp(0, 1)),
                "prediction_match":float((1 - kl_sub / kl_orig).clamp(0, 1)),
                "loss_moments":losses[1].tolist(),
            })
        loss = sum(v["loss_match"] for v in components) / 2
        prediction = sum(v["prediction_match"] for v in components) / 2
        measured.append({"hypothesis":name,"loss":loss,"prediction":prediction,
                         "product":loss * prediction * retain_proxy,"views":components})
    joint = min(s["product"] for s in measured[:len(hypotheses)])
    if per_view is not None:
        joint = (joint + measured[-1]["product"]) / 2
    return {"joint_proxy":joint,"retain_proxy":retain_proxy,
            "agreement":agreement,"control_kl":control_kl,"natural_crops":natural,"scenarios":measured}


def save_report(data, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path, action="append", default=[])
    parser.add_argument("--compare-scale", nargs=2, action="append", default=[])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--means", type=float, nargs=2, default=(7.0, 7.4))
    parser.add_argument("--teacher-weight", type=float, default=0.8)
    parser.add_argument("--original-anchor-weight", type=float, default=12)
    parser.add_argument("--moment-weight", type=float, default=1.0)
    parser.add_argument("--retained-crops", action="store_true")
    parser.add_argument("--confident-controls", action="store_true")
    parser.add_argument("--prediction-ensemble-spread", type=float, default=0)
    parser.add_argument("--restore-record",action="store_true")
    parser.add_argument("--class-prior", type=float, default=0)
    parser.add_argument("--balance-gradients", action="store_true")
    parser.add_argument("--full-kl-fraction", type=float, default=0)
    parser.add_argument("--class-aware-loss", action="store_true")
    parser.add_argument("--alternative-sharpness", type=float, default=1)
    parser.add_argument("--evaluation-base", choices=("prob","logit"), default="prob")
    parser.add_argument("--evaluation-sharpness", type=float, default=1)
    parser.add_argument("--lower-moments", action="store_true")
    parser.add_argument("--loss-model",type=Path)
    parser.add_argument("--source-view-fraction",type=float,default=0)
    parser.add_argument("--target-fraction", type=float, default=0.85)
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root))
    import unlearn as recipe
    assert args.epochs >= 0 and 0 <= args.teacher_weight <= 1 and 0 < args.target_fraction <= 1
    torch.manual_seed(615)
    raw = torch.load(root / "cache/codex_pool4/raw3_K64_s0.75_seed0.pt",
                     map_location="cpu", weights_only=True)
    labels = raw["y"].long().to(recipe.DEV)
    fit, validation = split_images(labels)
    cache = root / "cache/three_terms"
    fit_teacher = teacher_from_seeds(cache, (0, 1), labels)
    validation_teacher = teacher_from_seeds(cache, (2, 3), labels)
    full = torch.load(cache / "cibles_alternatives_p2_64.pt", map_location="cpu", weights_only=True)
    assert torch.equal(full["y"], raw["y"])
    full_teacher = full["za"].to(recipe.DEV)
    assert torch.allclose((full_teacher / 2).softmax(1), ((fit_teacher/2).softmax(1) +
                          (validation_teacher/2).softmax(1))/2, atol=2e-6)
    if args.evaluation_base == "logit":
        validation_teacher = teacher_from_seeds(cache, (2,3), labels, "logit")
    assert args.evaluation_sharpness > 0
    validation_teacher = args.evaluation_sharpness * validation_teacher
    prior_offset = None
    if args.class_prior:
        prior = torch.load(cache / "prior10.pt", map_location="cpu", weights_only=True)
        assert torch.equal(raw["y"], prior["y"])
        offset = prior["za"] - raw["za"]
        prior_offset = offset.mean(0).to(recipe.DEV)
        assert torch.allclose(offset, prior_offset.cpu().expand_as(offset), atol=1e-5)
        fit_teacher = fit_teacher + args.class_prior * prior_offset
        full_teacher = full_teacher + args.class_prior * prior_offset
    assert args.alternative_sharpness > 0
    fit_teacher = args.alternative_sharpness * fit_teacher
    full_teacher = args.alternative_sharpness * full_teacher
    def desired(means, cap, ranking=raw):
        recipe.set_shape(argparse.Namespace(target_shape="mix", gold_acc=0.15, correct_pow=8.0,
            target_type="rank", base="logit", ens_gamma=1.0, tail="exp", max_loss=cap,
            wrong_temp=1.0, class_aware=""))
        return [recipe.make_target_logits(ranking, v, 1.5)[1].to(recipe.DEV) for v in means]
    ranking = raw
    if args.class_aware_loss:
        ranking = torch.load(root / "cache/codex_pool4_structured/raw3_K64_s0.75_seed0.pt",
                             map_location="cpu", weights_only=True)
        assert torch.equal(ranking["y"],raw["y"])
    assert 0 <= args.full_kl_fraction <= 1
    train_targets = desired(args.means, 14 if args.means[0] <= 7 else 20, ranking)
    hypotheses = []
    target_cases = ([("lower",(5.5,5.5),14),("moderate",(7.0,7.4),14)] if args.lower_moments
                    else [("moderate",(7.0,7.4),14),("strong",(9.0,9.4),20)])
    for name, means, cap in target_cases:
        a, b = desired(means, cap)
        hypotheses.append((name, full_target(validation_teacher, labels, a),
                           full_target(validation_teacher, labels, b)))
        if prior_offset is not None:
            tempered = validation_teacher + prior_offset
            hypotheses.append((name + "_prior", full_target(tempered, labels, a),
                               full_target(tempered, labels, b)))
    images = recipe.load_images(root.parent / "forget_images.npz")
    original = recipe.load_model(root.parent / "model.pt").requires_grad_(False)
    source = recipe.load_model(args.source).requires_grad_(False)
    loss_model = (source if args.restore_record else
                  recipe.load_model(args.loss_model).requires_grad_(False) if args.loss_model else None)
    assert 0 <= args.source_view_fraction <= 1
    assert 0 <= args.prediction_ensemble_spread < 1
    assert not args.confident_controls or args.retained_crops
    fixed = fixed_views(recipe, original, images, validation, 87251)
    per_view = (per_view_targets(recipe,source,loss_model,fixed,labels,args.target_fraction)
                if loss_model is not None else None)
    training_crops = validation_crops = crop_controls = None
    bank_stats = {}
    if args.retained_crops:
        training_crops, bank_stats["fit"] = retained_crops(
            recipe, original, images, labels, fit, 67103, 3, 40, args.confident_controls)
        validation_crops, bank_stats["validation"] = retained_crops(
            recipe, original, images, labels, validation, 29119, 3, 12, args.confident_controls)
        if args.confident_controls:
            strengthened, bank_stats["strengthening"] = strengthen_crop_labels(
                recipe, original, training_crops)
            training_crops = (torch.cat((training_crops,strengthened)) if args.restore_record
                              else strengthened)
        crop_controls = crop_reference(recipe, original, validation_crops)
    baseline = measure(recipe, source, fixed, labels, hypotheses, crop_controls, per_view=per_view)
    kwargs = {"teacher_weight":args.teacher_weight,"target_fraction":args.target_fraction,
              "moment_weight":args.moment_weight,"balance_gradients":args.balance_gradients,
              "full_kl_fraction":args.full_kl_fraction,
              "source_view_fraction":args.source_view_fraction,
              "prediction_ensemble_spread":args.prediction_ensemble_spread,
              "source_anchor_weight":0.0 if args.restore_record else 4.0,
              "original_anchor_weight":args.original_anchor_weight}
    report = {"source":args.source.name,"fit_images":len(fit),"validation_images":len(validation),
              "proxy":"synthetic controls and gold targets; actual reference and control labels are hidden",
              "baseline":baseline,"settings":dict(kwargs,means=args.means,class_prior=args.class_prior,
              retained_crops=args.retained_crops,confident_controls=args.confident_controls,
              restore_record=args.restore_record,
              class_aware_loss=args.class_aware_loss,
              alternative_sharpness=args.alternative_sharpness,evaluation_base=args.evaluation_base,
              evaluation_sharpness=args.evaluation_sharpness,lower_moments=args.lower_moments,
              loss_model=str(args.loss_model) if args.loss_model else None),
              "crop_bank":bank_stats,"comparison":{},"epochs":[]}
    path = args.output.with_suffix(".json")
    save_report(report, path)
    print("SOURCE", args.source.name, baseline["joint_proxy"], flush=True)
    comparisons = [(p, 1.0) for p in args.compare]
    comparisons.extend((Path(name),float(scale)) for name,scale in args.compare_scale)
    for candidate, scale in comparisons:
        model = recipe.load_model(candidate).requires_grad_(False)
        metrics = measure(recipe, model, fixed, labels, hypotheses, crop_controls, scale, per_view)
        key = candidate.name if scale == 1 else f"{candidate.name}@{scale}"
        report["comparison"][key] = metrics
        print("COMPARE",key,metrics["joint_proxy"],metrics["retain_proxy"],flush=True)
        del model
    save_report(report, path)
    if not args.epochs:
        return
    best_epoch = 0
    def selection_value(metrics):
        return (metrics["scenarios"][-1]["product"] if args.restore_record else metrics["joint_proxy"])
    best_value = selection_value(baseline)
    def assess(epoch, model):
        nonlocal best_epoch, best_value
        metrics = measure(recipe, model, fixed, labels, hypotheses, crop_controls, per_view=per_view)
        report["epochs"].append({"epoch":epoch, **metrics})
        # Choose on the product across all three components and both target assumptions.
        if selection_value(metrics) > best_value:
            best_value, best_epoch = selection_value(metrics), epoch
        save_report(report, path)
        print("EPOCH", round(epoch,3),metrics["joint_proxy"],metrics["retain_proxy"],
              [s["prediction"] for s in metrics["scenarios"]], flush=True)
    trial = train(recipe,args.source,source,original,images,labels,train_targets,fit,args.epochs,
                  assess,fit_teacher,retained_views=training_crops,loss_reference=loss_model,**kwargs)
    del trial, fixed
    torch.cuda.empty_cache()
    report["selected_epoch"] = best_epoch
    save_report(report,path)
    if not best_epoch:
        print("Source leads joint validation",flush=True)
        return
    final = train(recipe,args.source,source,original,images,labels,train_targets,
                      torch.arange(len(images),device=labels.device),best_epoch,None,full_teacher,
                      retained_views=training_crops,loss_reference=loss_model,**kwargs)
    fresh = fixed_views(recipe,original,images,validation,37471)
    fresh_targets = (per_view_targets(recipe,source,loss_model,fresh,labels,args.target_fraction)
                     if loss_model is not None else None)
    report["fresh_source"] = measure(recipe,source,fresh,labels,hypotheses,crop_controls,per_view=fresh_targets)
    report["fresh_final"] = measure(recipe,final,fresh,labels,hypotheses,crop_controls,per_view=fresh_targets)
    save_report(report,path)
    model_tmp=args.output.with_name(args.output.stem+".partial.pt")
    recipe.save_checked(final,str(model_tmp))
    os.replace(model_tmp,args.output)
    print("READY",args.output,report["fresh_final"]["joint_proxy"],flush=True)


if __name__ == "__main__":
    main()
