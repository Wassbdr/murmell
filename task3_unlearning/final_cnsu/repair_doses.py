"""Interpolate the complete repair against official champion 4112; test the 500 development sources."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cnsu_phase_a import load_state, probability, kl, wrong_class_kl, digest

HOME = Path(".")
ROOT = Path("./work/codex_t3")
OUT = ROOT/"cnsu/doses"
sys.path.insert(0,str(HOME))
import task_template as tt
from t3_consist import alter


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--inverse4205",action="store_true")
    inverse=cli.parse_args().inverse4205
    work=OUT/"inverse4205" if inverse else OUT
    work.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.monotonic()
    anchor = load_state(ROOT/"consensus/codex_marge_isolee.pt")
    end = load_state(OUT/"repair100.pt")
    assert digest(OUT/"repair100.pt") == "57377ff2f5837b9c79ca3f7542aef847c6b28e1caa2022735d56a5d024c79217"
    changes = [k for k in anchor if not torch.equal(anchor[k],end[k])]
    assert all(k.startswith(("blocks.10.","blocks.11.","head.","norm.")) for k in changes),changes
    changed_cat = []
    rival = {}
    if inverse:
        rival_path=ROOT/"after4112/codex_4112_par_classe_moderee.pt"
        assert digest(rival_path)=="264602264901b9732e174bf93b5165e271648d627ca0e73f88ac96aa822d7ac4"
        rival=load_state(rival_path)
        changed_cat=[k for k in anchor if not torch.equal(anchor[k],rival[k])]
        assert all(k.startswith("head.") for k in changed_cat),changed_cat
    original = tt.build_model().cuda().eval().requires_grad_(False)
    original.load_state_dict(load_state(HOME/"model.pt"),strict=True)
    model = tt.build_model().cuda().eval().requires_grad_(False)
    model.load_state_dict(anchor,strict=True)
    target = load_state(ROOT/"cnsu/raw3_K64_s0.75_seed0.pt")
    kimi = load_state(ROOT/"cnsu/kimi_pg.pt")
    y = target["y"].long()
    assert torch.equal(y,kimi["y"].long())
    p = target["p2"].float(); p /= p.sum(1,keepdim=True)
    k = kimi["pg"].float(); k /= k.sum(1,keepdim=True)
    raw = tt.load_forget_images(HOME/"forget_images.npz")
    gen = np.load(HOME/"gen/gen_test.npz")
    si = torch.from_numpy(gen["images"][:650].copy())
    sy = torch.from_numpy(gen["cls"][:650].copy()).long()
    rng = torch.Generator().manual_seed(79623)
    torch.randperm(2500,generator=rng)
    ids = 2500+torch.randperm(500,generator=rng)
    assert set(ids.tolist()) == set(range(2500,3000))

    def tail(h):
        with torch.autocast("cuda", dtype=torch.float16):
            for b in model.blocks[-2:]:
                h = b(h)
            return model.forward_head(model.norm(h)).float()

    prepared = {}
    with torch.no_grad():
        for view in ("exact","alt","strong","retain"):
            hlist,qlist,klist,ys = [],[],[],[]
            g = torch.Generator(device="cuda").manual_seed(64202 if view=="alt" else 81399)
            for i in (torch.arange(384,650) if view=="retain" else ids).split(24):
                x = (si if view=="retain" else raw)[i].cuda().permute(0,3,1,2).float()/255
                if view in ("alt","strong"):
                    x = alter(x,g,.7 if view=="alt" else 1.)
                x = 2*x-1
                with torch.autocast("cuda",dtype=torch.float16):
                    z = original(x).float()
                    h = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
                    for block in model.blocks[:-2]:
                        h = block(h)
                    if not hlist and view=="exact":
                        assert float((tail(h)-model(x)).abs().max()) < .01
                if view=="retain":
                    label = sy[i].cuda()
                    q = probability(z)
                else:
                    label = y[i].cuda()
                    q = p[i].cuda()
                    if view in ("alt","strong"):
                        ce0 = F.cross_entropy(z,label,reduction="none")
                        ces = F.cross_entropy(2*q.clamp_min(1e-12).log(),label,reduction="none")
                        q = torch.where((ce0>ces)[:,None],probability(z),q)
                    klist.append(k[i].cuda())
                hlist.append(h);qlist.append(q);ys.append(label)
            prepared[view] = (torch.cat(hlist),torch.cat(qlist),torch.cat(klist) if klist else None,torch.cat(ys))
    del original,raw,gen,si
    report = dict(anchor_sha=digest(ROOT/"consensus/codex_marge_isolee.pt"),
                  repaired_sha=digest(OUT/"repair100.pt"),changed_keys=changes,
                  train_n=0,forget_dev_ids=ids.tolist(),retain_ids=list(range(384,650)),rows={})
    def make_state(repair, reverse):
        state = {}
        for key,weight in anchor.items():
            value = weight
            if key in changes and repair:
                value = value+repair*(end[key]-weight)
            if key in changed_cat and reverse:
                value = value+reverse*(weight-rival[key])
            state[key]=value
        return state
    if inverse:
        grid=[(r,a) for r in (0.,.5) for a in (0.,.5,1.,2.,3.)]
    else:
        grid=[(r,0.) for r in (0.,.5,1.,1.55,2.)]
    dev_loss = {}
    references = {}
    choices = []
    for dose,reverse in grid:
        state = make_state(dose,reverse)
        if dose==1. and not inverse: state=end
        model.load_state_dict(state,strict=True)
        assert len(state)==176 and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
        entry = {}
        if not inverse and dose in (.5,1.55,2.):
            fp = OUT/("repair_dose"+str(round(dose*100)).zfill(3)+".pt")
            torch.save(state,fp)
            tt.build_model().load_state_dict(load_state(fp),strict=True)
            assert fp.stat().st_size<450_000_000
            entry.update(file=fp.name,sha256=digest(fp),size_bytes=fp.stat().st_size)
        source_losses = {}
        for view,(x,q,kk,yv) in prepared.items():
            with torch.no_grad():
                z = torch.cat([tail(xx) for xx in x.split(32)])
                label = yv>=0
                ce = F.cross_entropy(z[label],yv[label],reduction="none")
                s = dict(kl=kl(z,q).mean().item(),ce=ce.mean().item(),ce_std=ce.std(unbiased=False).item(),
                         agreement=(z.argmax(1)==q.argmax(1)).float().mean().item(),
                         correct=int((z[label].argmax(1)==yv[label]).sum()),total_labeled=int(label.sum()))
                if kk is not None:
                    source_losses[view] = torch.stack([kl(z,q),kl(z,kk),wrong_class_kl(z,q,yv),wrong_class_kl(z,kk,yv)]).cpu().numpy()
                    s.update(kimi=kl(z,kk).mean().item(), wrong_p2=wrong_class_kl(z,q,yv).mean().item(),
                             wrong_kimi=wrong_class_kl(z,kk,yv).mean().item())
            entry[view]=s
        tag = f"repair{dose}_inverse{reverse}" if inverse else str(dose)
        if inverse and not reverse:
            references[dose] = (entry,source_losses)
        if inverse and reverse:
            ref,rl = references[dose]
            deltas = np.stack([rl[view] - source_losses[view] for view in ("exact","alt","strong")])
            rng = np.random.default_rng(74312)
            bs = rng.integers(500,size=(1500,500))
            avg = deltas.mean(2)
            # Paired bootstrap on source IDs, retaining exact and altered observations together.
            ci = np.quantile(deltas[:,:,bs].mean(3),[.025,.975],axis=2)
            against_anchor = report["rows"]["repair0.0_inverse0.0"] if dose or reverse else entry
            max_moment = max(abs(entry[v][q]-ref[v][q]) for v in ("exact","alt") for q in ("ce","ce_std"))
            max_absolute_moment = max(abs(entry[v][q]-against_anchor[v][q]) for v in ("exact","alt") for q in ("ce","ce_std"))
            eligible = bool((avg[:2,:2]>0).all() and avg[2,:2].min()>=-.0002 and
                            ci[0,:2,:2].min()>=-.0002 and np.min(avg[:2,2:])>=-.0001 and
                            max_moment<=.035 and max_absolute_moment<=.075 and
                            entry["retain"]["kl"]-against_anchor["retain"]["kl"] <= .001 and
                            entry["retain"]["agreement"]>=against_anchor["retain"]["agreement"]-2/266 and
                            entry["retain"]["correct"]>=against_anchor["retain"]["correct"]-2)
            entry["matched_comparison"] = dict(matched_dose=dose,gain_by_view_and_target=avg.tolist(),
                 ci95=ci.tolist(),maximum_moment_shift_vs_match=max_moment,
                 maximum_moment_shift_vs_anchor=max_absolute_moment,eligible=eligible)
            if eligible: choices.append((avg[:2,:2].mean(),dose,reverse,tag))
            print("comparison",tag,avg[:2,:2].tolist(),"strong",avg[2,:2].tolist(),"eligible",eligible,flush=True)
        report["rows"][tag]=entry
        print(dose,entry.get("sha256","reference"),entry["exact"]["kl"],entry["alt"]["kl"],entry["retain"]["kl"],flush=True)
    report.update(gpu=torch.cuda.get_device_name(),peak_mib=torch.cuda.max_memory_allocated()/2**20,
                  duration_s=time.monotonic()-t0)
    if inverse:
        report["changed_4205_keys"] = changed_cat
        report["frozen"] = None
        if choices:
            score,dose,reverse,tag = max(choices)
            state = make_state(dose,reverse)
            model.load_state_dict(state,strict=True)
            assert len(state)==176 and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
            dest=work/"inverse4205_selected.pt"
            torch.save(state,dest)
            tt.build_model().load_state_dict(load_state(dest),strict=True)
            report["frozen"]=dict(tag=tag,sha256=digest(dest),bytes=dest.stat().st_size)
        print("FROZEN",report["frozen"],flush=True)
    (work/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
