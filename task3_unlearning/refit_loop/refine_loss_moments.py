"""Refine a scored model against both competition loss moments and class probabilities."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from calibrate_losses import split_images


def conditional_kl(recipe, source, logits, labels):
    mask = F.one_hot(labels, source.shape[1]).bool()
    return recipe.kl_T(source.masked_fill(mask, -1000), logits.masked_fill(mask, -1000)).mean()


def moments(losses):
    return torch.stack((losses.mean(), losses.std(unbiased=False)))


def full_target(conditional_logits, labels, desired_loss):
    z = conditional_logits.double().clone()
    row = torch.arange(len(labels), device=z.device)
    z[row, labels] = -torch.inf
    loss = desired_loss.double().clamp(1e-5, 20)
    z[row, labels] = z.logsumexp(1) - loss - torch.log(-torch.expm1(-loss))
    return z.float()


def ensemble_target(conditional_logits, labels, desired_loss, spread):
    central = full_target(conditional_logits, labels, desired_loss)
    if not spread:
        return central
    members = [full_target(conditional_logits, labels, desired_loss * factor)
               for factor in (1 - spread, 1 + spread)]
    probabilities = 0.5 * (central / 2).softmax(1)
    for member in members:
        probabilities += 0.25 * (member / 2).softmax(1)
    return 2 * probabilities.clamp_min(1e-20).log()


def predict(recipe, model, images):
    with torch.autocast("cuda", dtype=torch.float16):
        return model(recipe.norm(images)).float()


def joint_gradient_weights(losses, head):
    """Choose weights from the three gradient directions on the shared head."""
    gradients = []
    for loss in losses:
        parts = torch.autograd.grad(loss, head, retain_graph=True)
        gradients.append(torch.cat([p.detach().float().reshape(-1) for p in parts]))
    matrix = torch.stack(gradients)
    gram = (matrix @ matrix.T).double().cpu().numpy()
    norms = np.sqrt(np.maximum(gram.diagonal(), 0))
    if not np.isfinite(norms).all() or norms.min() <= max(1e-10, 1e-5 * norms.max()):
        return [1.0] * len(losses)
    gram /= np.outer(norms, norms)
    weights = np.full(len(losses), 1 / len(losses))
    for _ in range(20):
        direction = -weights.copy()
        direction[np.argmin(gram @ weights)] += 1
        curvature = direction @ gram @ direction
        if curvature < 1e-12:
            break
        step = float(np.clip(-(direction @ gram @ weights) / curvature, 0, 1))
        if step < 1e-7:
            break
        weights += step * direction
    weights /= norms
    weights *= len(losses) / weights.sum()
    # The head is a proxy for two full transformer blocks. Keep all tasks active.
    return (0.25 + 0.75 * weights).tolist()


@torch.no_grad()
def fixed_validation(recipe, images, indices, original, source, seed=502, strength=0.5):
    gen = torch.Generator(device=images.device).manual_seed(seed)
    batches = []
    for ids in indices.split(48):
        clean = recipe.to01(images[ids])
        altered = recipe.augment(clean, strength, gen)
        crop = recipe.make_anchor(clean, gen, ["crop"])
        mosaic = recipe.make_anchor(clean, gen, ["mos2"])
        forget = torch.cat((clean, altered))
        anchors = torch.cat((crop, mosaic))
        batches.append((ids, forget, anchors, predict(recipe, source, forget),
                        predict(recipe, source, anchors), predict(recipe, original, anchors)))
    return batches


@torch.no_grad()
def evaluate(recipe, student, batches, labels, targets, alternative_teacher=None):
    clean, altered, target_clean, target_alt = [], [], [], []
    conditional, teacher_kl, shift, orig_anchor, agreement, weights = [], [], [], [], [], []
    for ids, forget, anchors, source_f, source_a, orig_a in batches:
        zf, za = predict(recipe, student, forget), predict(recipe, student, anchors)
        y = labels[ids].repeat(2)
        a, b = F.cross_entropy(zf, y, reduction="none").chunk(2)
        clean.append(a)
        altered.append(b)
        target_clean.append(targets[0][ids])
        target_alt.append(targets[1][ids])
        conditional.append(float(conditional_kl(recipe, source_f, zf, y)))
        if alternative_teacher is not None:
            ref = alternative_teacher[ids].repeat(2, 1)
            teacher_kl.append(float(conditional_kl(recipe, ref, zf, y)))
        shift.append(float(recipe.kl_T(source_a, za).mean()))
        orig_anchor.append(float(recipe.kl_T(orig_a, za).mean()))
        agreement.append(float((orig_a.argmax(1) == za.argmax(1)).float().mean()))
        weights.append(len(ids))
    def weighted(v):
        return sum(a * b for a, b in zip(v, weights)) / sum(weights)
    clean, altered = (moments(torch.cat(v)) for v in (clean, altered))
    tc, ta = (moments(torch.cat(v)) for v in (target_clean, target_alt))
    gap = float(((clean - tc).abs().sum() + (altered - ta).abs().sum()) / 2)
    result = dict(gap=gap, clean=clean.tolist(), altered=altered.tolist(),
                  alternative_kl=weighted(conditional), anchor_drift=weighted(shift),
                  original_anchor_kl=weighted(orig_anchor), anchor_agreement=weighted(agreement))
    if teacher_kl:
        result["spatial_teacher_kl"] = weighted(teacher_kl)
    return result


def train(recipe, source_path, source, original, images, labels, targets, indices, epochs, assess,
          alternative_teacher, teacher_weight, target_fraction, source_anchor_weight=8.0,
          original_anchor_weight=4.0, moment_weight=1.0, retained_views=None,
          balance_gradients=False, full_kl_fraction=0.0, loss_reference=None,
          source_view_fraction=0.0, prediction_ensemble_spread=0.0):
    student = recipe.load_model(source_path)
    params = recipe.select_params(student, "last2")
    optimizer = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator(device=images.device).manual_seed(1015)
    steps = math.ceil(len(indices) / 32)
    early_checks = {max(1, round(steps / 4)), max(1, round(steps / 2))}
    task_weights = (1.0, 1.0, 1.0)
    for epoch in range(1, math.ceil(epochs) + 1):
        perm = torch.randperm(len(indices), generator=gen, device=images.device)
        for step, ids in enumerate(indices[perm].split(32), start=1):
            if len(ids) == 1:
                continue
            x = recipe.to01(images[ids])
            with torch.no_grad():
                strength = 0.2 + 0.45 * torch.rand(len(x), generator=gen, device=x.device)
                altered = recipe.augment(x, strength, gen)
                anchor_ids = indices[torch.randint(len(indices), (48,), generator=gen, device=x.device)]
                anchors = recipe.make_anchor(recipe.to01(images[anchor_ids]), gen,
                                             ["crop", "mos2", "mos3", "mix", "cutmix"])
                if retained_views is not None:
                    take = len(anchors) // 2
                    choice = torch.randint(len(retained_views), (take,), generator=gen,
                                           device=images.device)
                    anchors = torch.cat((anchors[take:], retained_views[choice].float() / 255))
                forget = torch.cat((x, altered))
                source_f = predict(recipe, source, forget)
                reference_f = (source_f if loss_reference is source else
                               predict(recipe, loss_reference, forget) if loss_reference is not None else None)
                source_a = predict(recipe, source, anchors)
                orig_a = predict(recipe, original, anchors)
            logits = predict(recipe, student, torch.cat((forget, anchors)))
            zf, za = logits.split((len(forget), len(anchors)))
            y = labels[ids].repeat(2)
            losses = F.cross_entropy(zf, y, reduction="none")
            prior_loss = F.cross_entropy(source_f, y, reduction="none")
            target_loss = (F.cross_entropy(reference_f, y, reduction="none") if reference_f is not None
                           else torch.cat([t[ids] for t in targets]))
            target = target_fraction * target_loss + (1 - target_fraction) * prior_loss
            moment_loss = sum(F.smooth_l1_loss(moments(a), moments(b), beta=0.5)
                              for a, b in zip(losses.chunk(2), target.chunk(2)))
            sample_loss = F.smooth_l1_loss(losses, target)
            keep_alternatives = conditional_kl(recipe, source_f, zf, y)
            if alternative_teacher is not None:
                spatial = alternative_teacher[ids].repeat(2, 1)
                if source_view_fraction:
                    mask = F.one_hot(y,spatial.shape[1]).bool()
                    reference_prob = (spatial.masked_fill(mask,-1000)/2).softmax(1)
                    view_prob = (source_f.masked_fill(mask,-1000)/2).softmax(1)
                    mix = torch.lerp(reference_prob,view_prob,source_view_fraction)
                    spatial = 2 * mix.clamp_min(1e-14).log()
                spatial_loss = conditional_kl(recipe, spatial, zf, y)
                if full_kl_fraction:
                    target_logits = ensemble_target(spatial, y, target, prediction_ensemble_spread)
                    full_kl = recipe.kl_T(target_logits, zf).mean()
                    spatial_loss = (1 - full_kl_fraction) * spatial_loss + full_kl_fraction * full_kl
                keep_alternatives = (1 - teacher_weight) * keep_alternatives + teacher_weight * spatial_loss
            keep_anchors = recipe.kl_T(source_a, za).mean()
            restore_anchors = recipe.kl_T(orig_a, za).mean()
            tasks = (moment_weight * moment_loss + 0.15 * sample_loss, 10 * keep_alternatives,
                     source_anchor_weight * keep_anchors + original_anchor_weight * restore_anchors)
            if balance_gradients and (step - 1) % 6 == 0:
                task_weights = joint_gradient_weights(tasks, tuple(student.head.parameters()))
            loss = sum(weight * task for weight, task in zip(task_weights, tasks))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            if epoch == 1 and step in early_checks and assess is not None:
                assess(step / steps, student)
            if assess is None and (epoch - 1) * steps + step >= math.ceil(epochs * steps):
                print("Final training passes", epochs, flush=True)
                return student
        if assess is not None:
            assess(epoch, student)
        else:
            print("Final training epoch", epoch, flush=True)
    return student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--alternative-cache", type=Path)
    parser.add_argument("--teacher-weight", type=float, default=0.45)
    parser.add_argument("--target-fraction", type=float, default=0.65)
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root))
    import unlearn as recipe

    torch.manual_seed(615)
    raw = torch.load(root / "cache/codex_pool4/raw3_K64_s0.75_seed0.pt",
                     map_location="cpu", weights_only=True)
    recipe.set_shape(argparse.Namespace(target_shape="mix", gold_acc=0.15, correct_pow=8.0,
                                        target_type="rank", base="logit", ens_gamma=1.0,
                                        tail="exp", max_loss=14.0, wrong_temp=1.0, class_aware=""))
    targets = [recipe.make_target_logits(raw, mean, 1.5)[1].cuda() for mean in (7.0, 7.4)]
    labels = raw["y"].long().cuda()
    assert 0 <= args.teacher_weight <= 1 and 0 < args.target_fraction <= 1
    alternative_teacher = None
    if args.alternative_cache:
        aux = torch.load(args.alternative_cache, weights_only=True, map_location="cpu")
        assert torch.equal(aux["y"], raw["y"]) and aux["za"].shape == raw["za"].shape
        assert torch.isfinite(aux["za"]).all()
        alternative_teacher = aux["za"].cuda()
    images = recipe.load_images(root.parent / "forget_images.npz")
    original = recipe.load_model(root.parent / "model.pt").requires_grad_(False)
    source = recipe.load_model(args.source).requires_grad_(False)
    fit, valid = split_images(labels)
    if args.evaluate_only:
        student = recipe.load_model(args.output).requires_grad_(False)
        scores = []
        for seed, strength in ((8301, 0.3), (5167, 0.7)):
            fresh = fixed_validation(recipe, images, valid, original, source, seed, strength)
            parent = evaluate(recipe, source, fresh, labels, targets, alternative_teacher)
            child = evaluate(recipe, student, fresh, labels, targets, alternative_teacher)
            scores.append(dict(seed=seed, strength=strength, parent=parent, student=child,
                               gap_reduction=1 - child["gap"] / parent["gap"]))
            del fresh
        args.output.with_suffix(".views.json").write_text(json.dumps(scores, indent=2) + "\n")
        print("New views after full retraining", json.dumps(scores), flush=True)
        return
    fixed = fixed_validation(recipe, images, valid, original, source)
    baseline = evaluate(recipe, source, fixed, labels, targets, alternative_teacher)
    best, best_epoch, results = 0.0, 0, []
    print("Source", json.dumps(baseline), flush=True)

    def assess(epoch, model):
        nonlocal best, best_epoch
        metrics = evaluate(recipe, model, fixed, labels, targets, alternative_teacher)
        gain = 1 - metrics["gap"] / baseline["gap"]
        permitted = (metrics["anchor_agreement"] >= baseline["anchor_agreement"] - 0.03
                     and metrics["original_anchor_kl"] <= baseline["original_anchor_kl"] + 0.05
                     and metrics["alternative_kl"] <= 0.07)
        value = gain - metrics["anchor_drift"] - metrics["alternative_kl"]
        if alternative_teacher is not None:
            relative_kl_gain = 1 - metrics["spatial_teacher_kl"] / baseline["spatial_teacher_kl"]
            value = 0.75 * gain + 0.25 * relative_kl_gain - metrics["anchor_drift"]
            permitted = (permitted and relative_kl_gain > 0)
        results.append(dict(epoch=epoch, gain=gain, accepted=permitted, **metrics))
        print("Held-out epoch", epoch, json.dumps(results[-1]), flush=True)
        if permitted and value > best:
            best, best_epoch = value, epoch

    trial = train(recipe, args.source, source, original, images, labels, targets, fit, 5, assess,
                  alternative_teacher, args.teacher_weight, args.target_fraction)
    del trial, fixed
    torch.cuda.empty_cache()
    report = dict(source=str(args.source), validation_images=len(valid), baseline=baseline,
                  validation=results, selected_epoch=best_epoch, target_fraction=args.target_fraction,
                  alternative_cache=str(args.alternative_cache) if args.alternative_cache else None)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if best_epoch:
        final = train(recipe, args.source, source, original, images, labels, targets,
                      torch.arange(len(images), device=images.device), best_epoch, None,
                      alternative_teacher, args.teacher_weight, args.target_fraction)
        recipe.save_checked(final, str(args.output))
    else:
        print("No held-out improvement within the measured retention bounds.", flush=True)


if __name__ == "__main__":
    main()
