"""Rank trained models across plausible hidden metrics using completed public receipts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from compare_public_checkpoints import gold_distribution, group_outputs, log_softmax


def candidate_retention(outputs, original, soft_accuracy=False):
    candidates = []
    for ids in ((3, 4, 5), (3, 5), (5,)):
        logits = np.concatenate([original[i] for i in ids])
        old = log_softmax(logits / 2)
        posterior = np.exp(log_softmax(logits))
        confidence = posterior.max(1)
        labels = logits.argmax(1)
        new = np.stack([np.concatenate([model[i] for i in ids]) for model in outputs])
        distances = (np.exp(old) * (old - log_softmax(new / 2))).sum(2)
        estimated_classes = new.argmax(2)
        agreement = estimated_classes == labels
        if soft_accuracy:
            expected_accuracy = posterior[np.arange(len(labels))[None,:],estimated_classes]
        for quantile in (0, .65):
            keep = confidence >= np.quantile(confidence, quantile)
            sim = 1 / (1 + distances[:, keep].mean(1))
            candidates.extend([sim, sim * agreement[:, keep].mean(1)])
            if soft_accuracy:
                ratio = expected_accuracy[:,keep].mean(1)/max(confidence[keep].mean(),1e-9)
                candidates.append(sim*np.clip(ratio,0,1))
    return np.stack(candidates)


def candidate_moments(stats):
    choices = []
    for mean in (3.5, 4.5, 5.5, 6.5, 7.5, 9.):
        for std in (2.5, 3.5, 4.5, 5.5, 6.5):
            for mean_shift in (-1.5, -.5, .5):
                for std_shift in (-1., 0., 1.):
                    exact = np.abs(stats[0] - [mean, std]).sum(1)
                    altered = np.abs(stats[1] - [mean + mean_shift, std + std_shift]).sum(1)
                    a = np.clip(1 - exact / max(exact[0], 1e-6), 0, 1)
                    b = np.clip(1 - altered / max(altered[0], 1e-6), 0, 1)
                    choices.append((a + b) / 2)
    return np.stack(choices)


def candidate_predictions(pack, raw, prior, groups):
    labels = pack["labels"].numpy()
    idx = pack["ids"].numpy()
    offsets = (prior["za"] - raw["za"]).mean(0).numpy()
    cases = list(pack["loss_targets"].values())
    candidates = []
    for base in (raw["za"], 2 * raw["p2"].clamp_min(1e-12).log()):
        for sharpness in (.85, 1.5, 2.25):
            for prior_strength in (0., .65):
                alt = sharpness * base.numpy()[idx] + prior_strength * offsets
                for pair in cases:
                    for dispersion in (0, .55):
                        pred = []
                        for view, (target, lp) in enumerate(zip(pair, groups)):
                            ce = target.numpy()
                            q, entropy = gold_distribution(alt, labels, ce)
                            if dispersion:
                                q *= .5
                                for factor in (1 - dispersion, 1 + dispersion):
                                    alt_q, _ = gold_distribution(alt, labels, ce * factor)
                                    q += .25 * alt_q
                                entropy = float((q * np.log(q.clip(1e-15))).sum(1).mean())
                            # The official gold distribution averages probabilities at T=2.
                            targets = q if view == 0 else np.concatenate([q, q])
                            kl = np.maximum(entropy - np.einsum("kij,ij->k", lp, targets) /
                                            len(targets), 0)
                            pred.append(np.clip(1 - kl / max(kl[0], 1e-6), 0, 1))
                        candidates.append(np.mean(pred, axis=0))
    return np.stack(candidates)


def fitted_ensemble(moments, predictions, retention, scores, train, pair_ids=None, trajectory_weight=0.):
    # Hold out each training receipt in turn when choosing metric assumptions.
    models = len(scores)
    best_loss = np.empty(0)
    best_output = np.empty((0, models))
    best_indices = np.empty((0,3),dtype=int)
    target = scores[train]
    predictions_index, retention_index = np.indices((len(predictions),len(retention)))
    for index, loss in enumerate(moments):
        features = (loss * predictions[:,None,:] * retention[None,:,:]).reshape(-1, models)
        x = features[:,train]
        xy = x * target
        xx = np.square(x)
        denom = xx.sum(1, keepdims=True)
        slopes = (xy.sum(1, keepdims=True) - xy) / np.maximum(denom - xx, 1e-9)
        residual = np.abs(x * slopes - target)
        error = .65 * residual.mean(1) + .35 * np.quantile(residual, .85, axis=1)
        scale = xy.sum(1, keepdims=True) / np.maximum(denom, 1e-9)
        if pair_ids is not None and trajectory_weight:
            later, earlier = pair_ids
            estimated = scale * (features[:,later]-features[:,earlier])
            absolute_error = np.abs(estimated-(scores[later]-scores[earlier])).mean(axis=1)
            error = (1.-trajectory_weight)*error + trajectory_weight*absolute_error
        full = np.concatenate((best_output, features * scale))
        errors = np.concatenate((best_loss, error))
        new_indices = np.stack((np.full(features.shape[0],index),
                                predictions_index.ravel(),retention_index.ravel()),axis=1)
        indices = np.concatenate((best_indices,new_indices))
        keep = np.argpartition(errors, min(511, len(errors)-1))[:512]
        best_loss, best_output = errors[keep], full[keep]
        best_indices = indices[keep]
    return best_output, best_indices


def main():
    parser = argparse.ArgumentParser()
    for key in ("collection","raw","prior","receipts","output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--head-grid",action="store_true")
    parser.add_argument("--soft-control-accuracy",action="store_true")
    parser.add_argument("--scale-parent",action="append",default=[])
    parser.add_argument("--mix-parents",nargs=2,action="append",default=[])
    parser.add_argument("--mix-amounts", nargs="+", type=float, default=(.25,.50,.75))
    parser.add_argument("--route-views", nargs=3, action="append", default=[])
    parser.add_argument("--retention-counterfactuals",action="store_true")
    parser.add_argument("--half",type=int,choices=(0,1))
    parser.add_argument("--record",default="lina_ES2.0")
    parser.add_argument("--trajectory-weight", type=float, default=0.)
    parser.add_argument("--feedback-grid",action="store_true")
    args = parser.parse_args()
    if not 0 <= args.trajectory_weight <= 1:
        parser.error("Trajectory weight must be between zero and one")
    if not np.isfinite(args.mix_amounts).all():
        parser.error("Mixture coefficients must be finite")
    def load(path):
        return torch.load(path, weights_only=True, map_location="cpu")
    pack, raw, prior = (load(p) for p in (args.collection, args.raw, args.prior))
    scores = json.loads(args.receipts.read_text())
    if args.half is not None:
        selection = slice(args.half,None,2)
        for key in ("labels","ids"):
            pack[key] = pack[key][selection]
        pack["original"] = [v[selection] for v in pack["original"]]
        for group in ("models","loss_targets"):
            pack[group] = {k:[v[selection] for v in parts] for k,parts in pack[group].items()}
    def arrays(parts):
        return [value.float().numpy() for value in parts]
    original = arrays(pack["original"])
    models = {name:arrays(views) for name,views in pack["models"].items()}
    for scale in (.8, 1.5):
        models[f"w_m70a15_hs{scale}"] = [v * scale for v in models["w_m70a15"]]
    head_settings = {}
    if args.head_grid:
        freq = torch.bincount(raw["y"],minlength=100).float()
        freq = (freq/freq.max()).numpy()
        prior_bias = (prior["za"]-raw["za"]).mean(0).numpy()
        grid = ((.89,.89,0.),(.95,.95,0.),(1.07,1.07,0.),(1.16,1.16,0.),
                (1.28,1.28,0.),(1.10,.98,.2),(1.16,1.02,.4),
                (1.0,1.12,.2),(.96,1.08,0.),(1.08,.96,0.))
        for name in ("lina_ES2.0","codex_trois_trois_profs","codex_trois_controles"):
            if name not in models:
                continue
            for other,seen,offset in grid:
                scale = other+(seen-other)*freq
                key = f"{name}|{other:.2f},{seen:.2f},{offset:.2f}"
                head_settings[key] = {"parent":name,"other":other,"seen":seen,"offset":offset}
                models[key] = [v.mean(1,keepdims=True) + (v-v.mean(1,keepdims=True))*scale +
                               offset*prior_bias for v in models[name]]
    counterfactuals = {}
    components = {"original": original, **models}
    for clean, altered, retain in args.route_views:
        if any(part not in components for part in (clean, altered, retain)):
            raise ValueError("Each routed component requires measured logits")
        key = f"route|{clean}|{altered}|{retain}"
        models[key] = [components[clean][0], *components[altered][1:3], *components[retain][3:]]
        counterfactuals[key] = "Diagnostic using independently selected sources for exact images, altered images and controls."
    for left,right in args.mix_parents:
        if left not in models or right not in models:
            raise ValueError("Both mixture parents require measured logits")
        for amount in args.mix_amounts:
            key=f"mixture|{left}|{right}|{amount:.2f}"
            models[key]=[(1-amount)*a+amount*b for a,b in zip(models[left],models[right])]
            counterfactuals[key]="Mixture of measured logits. Weight interpolation requires separate evaluation."
    for parent in args.scale_parent:
        if parent not in models:
            raise ValueError(f"Missing measured logits for {parent}")
        for scale in (.84,.88,.92,.96,.98,1.02,1.04,1.08,1.12,1.16):
            key=f"temperature|{parent}|{scale:.2f}"
            models[key]=[scale*v for v in models[parent]]
            head_settings[key]={"parent":parent,"relative_scale":scale}
    if args.feedback_grid:
        for parent in ("codex_trois_controles","codex_trois_trois_profs",
                       "codex_restauration_affine_hors_pli"):
            if parent not in models:
                continue
            parent_scale = .89 if parent.endswith("hors_pli") else 1.
            for scale in (.64,.70,.74,.78,.81,.84,.87,.90,.93,.97):
                key = f"feedback|{parent}|{scale:.2f}"
                models[key] = [z*(scale/parent_scale) for z in models[parent]]
                head_settings[key] = {"parent":parent,"total_scale":scale,"parent_scale":parent_scale}
    if args.retention_counterfactuals:
        parent = "codex_trois_controles"
        corrected = [.89*part for part in models[parent][:3]]
        for name, controls in (("unscaled",models[parent][3:]),("record",models["lina_ES2.0"][3:]),
                               ("original",original[3:])):
            key = "counterfactual_controls_"+name
            models[key] = corrected + controls
            counterfactuals[key] = "Diagnostic using measured outputs from two different models."
        for fraction in (.25,.5,.75):
            key = f"counterfactual_controls_original_{fraction:.2f}"
            controls = [(1-fraction)*(.89*current)+fraction*old
                        for current,old in zip(models[parent][3:],original[3:])]
            models[key] = corrected+controls
            counterfactuals[key] = "Diagnostic with interpolated control logits and fixed forget logits."
    labels = pack["labels"].numpy()
    names = ["original", *models]
    outputs = [original, *models.values()]
    stats, probabilities = zip(*(group_outputs(outputs, group, labels) for group in ((0,), (1,2))))
    moments = candidate_moments(stats)
    predictions = candidate_predictions(pack, raw, prior, probabilities)
    retention = candidate_retention(outputs, original, args.soft_control_accuracy)
    observed = np.array([scores.get(n, [np.nan])[0] for n in names])
    prior_ids = np.array([scores.get(n, [None, 0])[1] for n in names])
    early = np.flatnonzero(np.isfinite(observed) & (prior_ids <= 1834))
    recent = np.flatnonzero(np.isfinite(observed) & (prior_ids > 1834))
    historical, _ = fitted_ensemble(moments, predictions, retention, observed, early)
    train = np.concatenate((early, recent))
    pairs = []
    if args.trajectory_weight:
        for idx in sorted(train, key=lambda n: prior_ids[n]):
            previous = train[prior_ids[train] < prior_ids[idx]]
            if prior_ids[idx] >= 2047 and len(previous):
                pairs.append((idx, previous[np.argmax(observed[previous])]))
    pair_ids = np.array(pairs[-8:]).T if pairs else None
    all_fit, selected = fitted_ensemble(moments, predictions, retention, observed,
                                        train, pair_ids, args.trajectory_weight)
    check = np.median(historical, axis=0)
    record = names.index(args.record)
    if not np.isfinite(observed[record]):
        raise ValueError(f"The reference needs an official receipt: {args.record}")
    changes = all_fit - all_fit[:,record,None]
    term_changes = {}
    for term,values,idx in zip(("loss","prediction","retention"),
                               (moments,predictions,retention),selected.T):
        powers = values[idx]
        term_changes[term] = np.median(powers-powers[:,record,None],axis=0)
    report = {"retrospective_checks":{names[i]:{"estimate":float(check[i]),
                  "official":float(observed[i])} for i in recent},
              "forward_mae":float(np.abs(check[recent] - observed[recent]).mean()),
              "combinations":len(moments)*len(predictions)*len(retention),
              "calibration_receipts":int(len(early)+len(recent)),
              "trajectory_weight":args.trajectory_weight,
              "trajectory_pairs":len(pairs[-8:]),
              "record_name":args.record,"record_official":float(observed[record]),
              "head_settings":head_settings,
              "counterfactuals":counterfactuals,
              "candidates":{n:{"median_change":float(np.median(changes[:,i])),
                  "fraction_ahead":float((changes[:,i]>0).mean()),
                  "component_changes":{k:float(v[i]) for k,v in term_changes.items()},
                  "range":np.quantile(changes[:,i],[.1,.9]).tolist()} for i,n in enumerate(names)}}
    temp=args.output.with_name(args.output.name+f".{os.getpid()}.partial")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temp.replace(args.output)
    print("Historical held-out MAE",round(report["forward_mae"],4),flush=True)
    for name, result in report["candidates"].items():
        if (name.startswith("codex_trois") and "|" not in name) or name=="lina_ES2.0":
            print(name, round(result["median_change"],4),round(result["fraction_ahead"],2),flush=True)


if __name__ == "__main__":
    main()
