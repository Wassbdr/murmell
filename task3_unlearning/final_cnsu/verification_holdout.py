"""Independent, source-paired external replay for the CNSU experiment on track 3."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path(os.getenv("CNSU_TRACK3_HOME", str(Path.home()/"t3")))
OUT = Path(os.getenv("CNSU_EXTERNAL_DIR","./work/experiments/cnsu/verification_external"))
sys.path.insert(0,str(HOME))
import task_template as tt
from t3_consist import alter
tt.DEVICE = "cpu"

ORIGINAL=HOME/"model.pt"
BEST=HOME/"codex_eval_moments/poids/codex_marge_isolee.pt"
PROXY=HOME/"kimi_pg.pt"
VIEWS64=OUT/"verification_raw3_K64_s0.75_seed0.pt"
F_PATH=HOME/"forget_images.npz"
GEN=HOME/"gen"
SEED=3108713
SCHEMA=3
PARENT_SHA="616ab7cbf452ab6ab6cbd841cdb9e0563d79f00f994c57c0a6bc71d962f24b52"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def hashes(images):
    return [hashlib.sha256(memoryview(x)).digest() for x in images]


def read(path):
    return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


def split():
    with np.load(F_PATH) as d:
        all_f=d["images"]
    fhash=hashes(all_f)
    ids=np.array([i for i in range(3000,len(all_f)) if fhash[i] not in set(fhash[:3000])])
    assert len(set(fhash[i] for i in ids)) == len(ids)
    with np.load(GEN/"gen_test.npz") as d:
        te,ty=d["images"],d["cls"]
    h=hashes(te)
    fit=set(h[:650])
    canonical=[]
    for i in range(650,min(852,len(te))):
        if h[i] not in fit:
            canonical.append(i)
            fit.add(h[i])
    teix=np.asarray(canonical)
    with np.load(GEN/"gen_tr1.npz") as d:
        tr,rl=d["images"],d["cls"]
    earlier=set(h)|set(fhash)
    unique={}
    for i,k in enumerate(hashes(tr)):
        if k not in earlier and k not in unique:
            unique[k]=i
    seen=sorted(unique)[-512:]
    ri=np.asarray([unique[k] for k in seen])
    assert len(ri)==512 and not set(seen)&earlier
    ids_y=read(PROXY)["y"][ids].long()
    arrays={"forget_exact":all_f[ids],"forget_altered":all_f[ids],"forget_novel":all_f[ids],"retain_held":te[teix],"retain_reserved":tr[ri],"retain_reserved_altered":tr[ri]}
    labels={"ids":torch.tensor(ids),"forget":ids_y,"retain_held":torch.tensor(ty[teix]).long(),"retain_reserved":torch.tensor(rl[ri]).long(),
            "retain_reserved_altered":torch.tensor(rl[ri]).long()}
    record={"forget_ids":ids.tolist(),"gen_test_indices":teix.tolist(),"gen_tr1_indices":ri.tolist(),"gen_tr1_source_hashes":[hh.hex() for hh in seen],
            "target_label_provenance":"Original argmax pseudo-labels, never official labels", "cnsu_splits":"forget train <2500, development 2500:3000, external 3000:3500. Retain sensitivity, trust, and development use only gen_test[:650].",
            "parent_exposure":"The 4112 lineage previously trained using generated tr1 images. These are independent of the CNSU fit. The 202 standard images were used for prior local diagnostics.",
            "views":"Official Task 3 preprocessing. Hidden alteration proxy is one fixed t3_consist.alter strength .7 run (seed 3108713; canonical batches of 48). Additional stress view uses a random affine crop, rescaling through 151 pixels and six-bit quantization. Stress retain uses strength .45 seed 3108756."}
    return arrays,labels,ids,record


def state_check(path):
    sd=read(path)
    if len(sd)!=176 or not all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in sd.values()):
        raise ValueError("A checkpoint must contain 176 finite float32 tensors")
    m=tt.build_model().cuda().eval()
    m.load_state_dict(sd,strict=True)
    return m


def forward(model,arrays):
    outputs={}
    for name,images in arrays.items():
        rows=[]
        strength=.7 if name=="forget_altered" else .45 if name=="retain_reserved_altered" else 0.
        rng=torch.Generator(device="cuda").manual_seed(SEED+(43 if name.startswith("retain") else 0))
        for i in range(0,len(images),48):
            x=tt.preprocess(torch.from_numpy(images[i:i+48])).cuda()
            if strength:
                x=alter((x+1)/2,rng,strength)*2-1
            elif name=="forget_novel":
                rr=np.random.default_rng(837513+i)
                count=len(x); scale=rr.uniform(.67,.92,count); ang=rr.uniform(-.12,.12,count); flips=rr.choice([-1.,1.],count)
                aff=np.stack([scale*np.cos(ang)*flips, -scale*np.sin(ang), rr.uniform(-1,1,count)*(1-scale),
                              scale*np.sin(ang)*flips, scale*np.cos(ang), rr.uniform(-1,1,count)*(1-scale)],axis=-1).reshape(-1,2,3)
                grid=F.affine_grid(torch.tensor(aff,dtype=torch.float32,device="cuda"),x.shape,align_corners=False)
                x=F.grid_sample(x,grid,padding_mode="reflection",mode="bilinear",align_corners=False)
                x=F.interpolate(x,(151,151),mode="bilinear",align_corners=False,antialias=True)
                x=F.interpolate(x,(224,224),mode="bicubic",align_corners=False).clamp(-1,1)
                x=((x+1)*31.5).round()/31.5-1
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                rows.append(model(x).float().cpu())
        outputs[name]=torch.cat(rows)
    return outputs


def cache(name,path,arrays):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",name):raise ValueError(name)
    uid=sha(path)
    if name=="4112": assert uid==PARENT_SHA
    if name=="original": assert uid=="485881a03ad064b5e6defc245ae3f45fd96714bf065ff96543a5eeca9f642571"
    dest=OUT/("logits_"+name+".pt")
    if dest.exists():
        payload=read(dest)
        if payload["sha256"]==uid and payload["seed"]==SEED and payload.get("schema")==SCHEMA:return payload
    torch.cuda.reset_peak_memory_stats()
    clock=time.monotonic()
    model=state_check(path)
    logits=forward(model,arrays)
    torch.cuda.synchronize()
    payload={"name":name,"path":str(path),"sha256":uid,"bytes":path.stat().st_size,"seed":SEED,"schema":SCHEMA,"outputs":logits,
             "runtime_seconds":time.monotonic()-clock,"peak_gpu_mib":torch.cuda.max_memory_allocated()/2**20}
    del model
    torch.save(payload,dest)
    return payload


def summarize(a):
    a=a.double()
    ii=torch.randint(len(a),(4000,len(a)),generator=torch.Generator().manual_seed(115270))
    return {"mean":float(a.mean()),"ci95":torch.quantile(a[ii].mean(1),torch.tensor([.025,.975],dtype=torch.double)).tolist()}


def kld(ref,logq):
    ref=ref.double()
    return (ref*(ref.clamp_min(1e-30).log()-logq)).sum(1)


def load_reference(path,key,ids):
    if path.suffix==".npz":
        with np.load(path) as data:q=torch.tensor(data[key])
    else:q=read(path)[key]
    q=q.double()
    if q.ndim==3:
        assert q.shape[0]==2 and q.shape[2]==100
        vals={k:q[i,ids] for i,k in enumerate(("forget_exact","forget_altered"))}
    else:
        assert q.shape==(3500,100)
        vals={k:q[ids] for k in ("forget_exact","forget_altered","forget_novel")}
    for name,v in vals.items():
        if v.min() < 0 or not torch.allclose(v.sum(1),torch.ones(len(v),dtype=v.dtype),atol=.02):
            raise ValueError("Distillation data must be a probability distribution")
        vals[name]=v/v.sum(1,keepdim=True)
    return vals


def metrics(payload,parent,base,labels,refs):
    zz=payload["outputs"]; anchor=base["outputs"]; teacher=parent["outputs"]
    result={"sha256":payload["sha256"],"bytes":payload["bytes"],"runtime_s":payload["runtime_seconds"],"peak_inference_mib":payload["peak_gpu_mib"],
            "compare_to":base["name"],"forget":{},"retain":{},"accuracy_gate":"Official 70% validation unavailable locally; generated control accuracy is reported separately."}
    for name,table in refs.items():
        delta=[]
        cond_deltas=[]
        result["forget"][name]={}
        for view,q in table.items():
            new=kld(q,(zz[view].double()/2).log_softmax(1))
            old=kld(q,(anchor[view].double()/2).log_softmax(1))
            delta.append(old-new)
            y=labels["forget"];ce=lambda z:F.cross_entropy(z.double(),y,reduction="none")
            new_ce=ce(zz[view]);old_ce=ce(anchor[view])
            mask=F.one_hot(y,100).bool()
            qr=q.masked_fill(mask,0); qr=qr/qr.sum(1,keepdim=True).clamp_min(1e-30)
            def log_other(z):
                return (z.double()/2).masked_fill(mask,-1e9).log_softmax(1)
            off_new=kld(qr,log_other(zz[view]));off_old=kld(qr,log_other(anchor[view]))
            cond_deltas.append(off_old-off_new)
            result["forget"][name][view]={"n":len(q),"target_kl":float(new.mean()),"kl_gain":summarize(old-new),
                "alternative_class_kl":float(off_new.mean()),"alternative_kl_gain":summarize(off_old-off_new),
                "ce_mean":float(new_ce.mean()),"ce_std":float(new_ce.std()),"ce_mean_delta":float((new_ce-old_ce).mean()),
                "ce_std_delta":float(new_ce.std()-old_ce.std()),
                "true_pseudolabel_top1":float((zz[view].argmax(1)==y).float().mean()),
                "mean_true_pseudolabel_probability":float(zz[view].double().softmax(1).gather(1,y[:,None]).mean()),
                "target_top1_agreement":float((zz[view].argmax(1)==q.argmax(1)).float().mean())}
        result["forget"][name]["paired_alternative_kl_gain"]=summarize((cond_deltas[0]+cond_deltas[1])/2)
        joint=(delta[0]+delta[1])/2
        result["forget"][name]["paired_exact_altered_kl_gain"]=summarize(joint)
        keep=labels["ids"]!=3031
        result["forget"][name]["external_3031_sensitivity"]={"n_without":int(keep.sum()),"gain_exact_without":float(delta[0][keep].mean()),"gain_altered_without":float(delta[1][keep].mean()),"gain_joint_without":float(joint[keep].mean()),"difference_from_full":float(joint[keep].mean()-joint.mean())}
    for view in ("retain_held","retain_reserved","retain_reserved_altered"):
        y=labels[view];mask=y>=0
        q=(teacher[view].double()/2).softmax(1)
        old=kld(q,(anchor[view].double()/2).log_softmax(1))
        new=kld(q,(zz[view].double()/2).log_softmax(1))
        orig=teacher[view].argmax(1); pred=zz[view].argmax(1); anc=anchor[view].argmax(1)
        acc=(pred[mask]==y[mask]).float();oldacc=(anc[mask]==y[mask]).float()
        agree=(pred==orig).float();prev=(anc==orig).float()
        result["retain"][view]={"n":len(y),"labeled":int(mask.sum()),"accuracy":float(acc.mean()),
            "accuracy_delta":summarize(acc-oldacc),"correct":int(acc.sum()),"original_correct":int((orig[mask]==y[mask]).sum()),
            "original_agreement":float(agree.mean()),"agreement_delta":summarize(agree-prev),
            "kl_from_original":float(new.mean()),"kl_gain":summarize(old-new)}
    return result


def parameter_change(path, anchor):
    newer=read(path); old=read(anchor)
    assert newer.keys()==old.keys()
    changed=[]
    sq=0.;parent_sq=0.
    for name,w in newer.items():
        parent=old[name]
        if not torch.equal(w,parent):
            changed.append(name)
            sq+=float(torch.linalg.vector_norm(w-parent))**2
        parent_sq+=float(torch.linalg.vector_norm(parent))**2
    return {"changed_tensors":changed,"changed_count":len(changed),"l2_norm":sq**.5,"relative_l2":(sq/max(parent_sq,1e-30))**.5}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",action="append",default=[],help="name=checkpoint path")
    parser.add_argument("--reference",action="append",default=[],help="alias=checkpoint-path:probability-key")
    parser.add_argument("--include-cached",action="store_true")
    parser.add_argument("--pair",action="append",default=[],help="Compare a preregistered candidate against a matched ablation, name,reference")
    parser.add_argument("--report-name", default=None, help="Store a separately preregistered comparison under a verification_ JSON filename")
    args=parser.parse_args()
    torch.set_num_threads(3)
    OUT.mkdir(parents=True,exist_ok=True)
    arrays,labels,ids,manifest=split()
    (OUT/"verification_split.json").write_text(json.dumps(manifest,indent=2))
    models={"original":ORIGINAL,"4112":BEST}
    models.update({p.split("=",1)[0]:Path(p.split("=",1)[1]) for p in args.model})
    pipe={name:cache(name,path,arrays) for name,path in models.items()}
    assert (pipe["original"]["outputs"]["forget_exact"].argmax(1)==labels["forget"]).double().mean()>.99
    if args.include_cached:
        for f in OUT.glob("logits_*.pt"):
            if f.stem[7:] not in pipe:
                p=read(f);pipe[p["name"]]=p
    specs={"team_pg":(PROXY,"pg")}
    for k in args.reference:
        nm,body=k.split("=",1);p,key=body.rsplit(":",1);specs[nm]=(Path(p),key)
    specs["views64"]=(VIEWS64,"p2")
    refs={name:load_reference(path,key,ids) for name,(path,key) in specs.items()}
    q=refs["views64"]["forget_exact"]
    refs["views64_fixed_p2"]=dict(refs["views64"])
    y=labels["forget"]
    masks={}
    for key in ("forget_altered","forget_novel"):
        orig_altered=pipe["original"]["outputs"][key].double()
        alt_mask=F.cross_entropy(orig_altered,y,reduction="none")>F.cross_entropy(2*q.clamp_min(1e-30).log(),y,reduction="none")
        refs["views64"][key]=torch.where(alt_mask[:,None],(orig_altered/2).softmax(1),q)
        masks[key]=int(alt_mask.sum())
    report={"ouachek_altered_replacements":masks,"primary_reference":"views64","references":{n:{"path":str(path),"key":key,"sha256":sha(path)} for n,(path,key) in specs.items()},
            "split":{k:manifest[k] for k in ("cnsu_splits","parent_exposure","views")},"results":{}}
    for key,p in pipe.items():
        report["results"][key]={"vs_4112":metrics(p,pipe["original"],pipe["4112"],labels,refs)}
        if key=="original" or key.lower().startswith("b"):
            report["results"][key]["vs_original"]=metrics(p,pipe["original"],pipe["original"],labels,refs)
    for name,p in pipe.items():
        start=ORIGINAL if name=="original" or name.lower().startswith("b") else BEST
        report["results"][name]["parameter_delta_from_start"]=parameter_change(Path(p["path"]),start)
    for pair in args.pair:
        name,ref=pair.split(",",1)
        report["results"][name]["vs_"+ref]=metrics(pipe[name],pipe["original"],pipe[ref],labels,refs)
    report_path=OUT/("verification_baselines.json" if set(pipe)=={"original","4112"} else "verification_results.json")
    if args.report_name:
        if re.fullmatch(r"verification_[A-Za-z0-9_]+[.]json",args.report_name) is None:
            raise ValueError(args.report_name)
        report_path=OUT/args.report_name
    tmp=report_path.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(report_path)
    print("Written",report_path,flush=True)
    for name in pipe:
        r=report["results"][name]["vs_4112"]
        print(name,"target",[(k,round(v["forget_exact"]["target_kl"],5),round(v["forget_altered"]["target_kl"],5)) for k,v in r["forget"].items()],
              "retain",[(k,round(v["kl_from_original"],5),v["correct"],v["labeled"],round(v["original_agreement"],4)) for k,v in r["retain"].items()],flush=True)

if __name__=="__main__":main()
