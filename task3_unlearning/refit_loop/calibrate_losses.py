"""Calibrate the head of a publicly scored model on forget-image loss targets."""

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def split_images(labels):
    generator = torch.Generator().manual_seed(713)
    train, valid = [], []
    for label in labels.cpu().unique():
        indices = (labels.cpu() == label).nonzero().flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = min(len(indices) - 1, max(1, round(len(indices) * 0.2)))
        train.append(indices[count:])
        valid.append(indices[:count])
    return [torch.cat(parts).to(labels.device) for parts in (train, valid)]


def extract(recipe, model, original, images, indices, seed):
    device = images.device
    gen = torch.Generator(device=device).manual_seed(seed)
    indices = indices[torch.randperm(len(indices), generator=gen, device=device)]
    batches = {view: [] for view in ("clean", "altered", "crop", "mosaic")}
    for batch in indices.split(48):
        clean = recipe.to01(images[batch])
        views = (clean, recipe.augment(clean, 0.5, gen),
                 recipe.make_anchor(clean, gen, ["crop"]),
                 recipe.make_anchor(clean, gen, ["mos2"]))
        for name, pixels in zip(batches, views):
            with torch.autocast("cuda", dtype=torch.float16):
                features = model.forward_head(model.forward_features(recipe.norm(pixels)),
                                              pre_logits=True)
                z = model.head(features).float()
                if name in ("crop", "mosaic"):
                    original_z = original(recipe.norm(pixels)).float()
                else:
                    original_z = z
            x = F.pad(features.float(), (0, 1), value=1)
            batches[name].append((x, z, original_z))
    return indices, {name: tuple(torch.cat(column) for column in zip(*rows))
                     for name, rows in batches.items()}


def fit(groups, labels, targets, extra_label_weight, penalty):
    clean_x, clean_z, _ = groups["clean"]
    alt_x, alt_z, _ = groups["altered"]
    x = torch.cat((clean_x, alt_x))
    z = torch.cat((clean_z, alt_z))
    y = labels.repeat(2)
    target_ce = torch.cat(targets).clamp(0.015, 14)
    alternatives = z.scatter(1, y[:, None], -torch.inf)
    desired_logit = (alternatives.logsumexp(1) - target_ce
                     - (-torch.expm1(-target_ce)).log())
    residual = (desired_logit - z.gather(1, y[:, None])[:, 0]).clamp(-13, 9)
    anchor = torch.cat((groups["crop"][0], groups["mosaic"][0]))
    gram = x.T @ x + 2 * (anchor.T @ anchor)
    gram.diagonal().add_(penalty * gram.diagonal().mean())
    correction = x.new_zeros((x.shape[1], z.shape[1]))
    for label in y.unique():
        selected = y == label
        xc = x[selected]
        normal = gram + extra_label_weight * (xc.T @ xc)
        rhs = (1 + extra_label_weight) * (xc.T @ residual[selected])
        correction[:, label] = torch.cholesky_solve(rhs[:, None],
                                                    torch.linalg.cholesky(normal))[:, 0]
    return correction


def evaluate(recipe, groups, labels, targets, correction):
    gap, alternatives, anchors, original_anchors, agreement, moments = [], [], [], [], [], {}
    for view, target in zip(("clean", "altered"), targets):
        x, source, _ = groups[view]
        z = source + x @ correction
        losses = F.cross_entropy(z, labels, reduction="none")
        moments[view] = [float(losses.mean()), float(losses.std())]
        gap.append(float((losses.mean() - target.mean()).abs()
                         + (losses.std() - target.std()).abs()))
        # Identical finite masks preserve normalized probabilities on the other classes.
        alternatives.append(float(recipe.kl_T(source.scatter(1, labels[:, None], -1000),
                                              z.scatter(1, labels[:, None], -1000)).mean()))
    for view in ("crop", "mosaic"):
        x, source, original = groups[view]
        z = source + x @ correction
        anchors.append(float(recipe.kl_T(source, z).mean()))
        original_anchors.append(float(recipe.kl_T(original, z).mean()))
        agreement.append(float((z.argmax(1) == original.argmax(1)).float().mean()))
    return dict(moment_gap=sum(gap) / 2, moments=moments, other_kl=max(alternatives),
                anchor_kl=max(anchors), original_anchor_kl=sum(original_anchors) / 2,
                original_agreement=sum(agreement) / 2)


@torch.inference_mode()
def main():
    parser = ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root))
    import unlearn as recipe

    torch.set_float32_matmul_precision("highest")
    raw = torch.load(root / "cache/codex_pool4/raw3_K64_s0.75_seed0.pt",
                     map_location="cpu", weights_only=True)
    recipe.set_shape(Namespace(target_shape="mix", gold_acc=0.15, correct_pow=8.0,
                               target_type="rank", base="logit", ens_gamma=1.0,
                               tail="exp", max_loss=14.0, wrong_temp=1.0, class_aware=""))
    targets = [recipe.make_target_logits(raw, mean, 1.5)[1].cuda() for mean in (7.0, 7.4)]
    labels = raw["y"].long().cuda()
    source = recipe.load_model(args.source)
    original = recipe.load_model(root.parent / "model.pt")
    images = recipe.load_images(root.parent / "forget_images.npz")
    train_ids, valid_ids = split_images(labels)
    train_ids, train = extract(recipe, source, original, images, train_ids, 807)
    valid_ids, valid = extract(recipe, source, original, images, valid_ids, 316)
    print(f"Training images {len(train_ids)}, independent validation images {len(valid_ids)}",
          flush=True)
    train_targets = [t[train_ids] for t in targets]
    valid_targets = [t[valid_ids] for t in targets]
    zero = torch.zeros((source.head.in_features + 1, source.head.out_features), device="cuda")
    baseline = evaluate(recipe, valid, labels[valid_ids], valid_targets, zero)
    print("Source on validation", json.dumps(baseline), flush=True)
    best, selected = 0.04, None
    for label_weight, penalty in ((8, 0.12), (24, 0.08), (48, 0.18)):
        delta = fit(train, labels[train_ids], train_targets, label_weight, penalty)
        for scale in (0.25, 0.45, 0.7, 1.0):
            metrics = evaluate(recipe, valid, labels[valid_ids], valid_targets, scale * delta)
            gain = 1 - metrics["moment_gap"] / max(baseline["moment_gap"], 1e-6)
            value = gain - metrics["anchor_kl"] - metrics["other_kl"]
            allowed = (metrics["other_kl"] <= 0.04 and metrics["anchor_kl"] <= 0.04
                       and metrics["original_anchor_kl"] <= baseline["original_anchor_kl"] + 0.04
                       and metrics["original_agreement"] >= baseline["original_agreement"] - 0.03)
            print("Validation", label_weight, penalty, scale, round(gain, 4),
                  round(metrics["anchor_kl"], 4), round(metrics["other_kl"], 4), allowed,
                  flush=True)
            if allowed and value > best:
                best, selected = value, (label_weight, penalty, scale, metrics)
    report = {"source": str(args.source), "validation_images": len(valid_ids),
              "source_validation": baseline, "selected": selected}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if selected is None:
        print("No setting improved held-out moment matching within preservation bounds.",
              flush=True)
        return
    label_weight, penalty, scale, _ = selected
    all_ids = torch.cat((train_ids, valid_ids))
    groups = {view: tuple(torch.cat((a, b)) for a, b in zip(train[view], valid[view]))
              for view in train}
    delta = scale * fit(groups, labels[all_ids], [t[all_ids] for t in targets],
                        label_weight, penalty)
    assert torch.isfinite(delta).all()
    source.head.weight.add_(delta[:-1].T)
    source.head.bias.add_(delta[-1])
    recipe.save_checked(source, str(args.output))
    print("Selected", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
