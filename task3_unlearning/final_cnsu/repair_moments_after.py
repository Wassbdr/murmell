"""Repair four forget loss moments in the head after the full A32 update."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from original_basis import ROOT,sha
import task_template as tt
from t3_consist import alter
from t3_fresh_holdout import OUT as FRESH

OUT=Path("./work/cnsu_repair")
def state(path):
    return torch.load(path,map_location="cpu",weights_only=True,mmap=True)
def model(sd):
    m=tt.build_model().cuda().eval();m.load_state_dict(sd,strict=True);return m
def stack_head(sd):
    return torch.cat([sd["head.weight"].T,sd["head.bias"].unsqueeze(0)]).cuda()
def prep(pixels):
    return pixels.cuda().permute(0,3,1,2).float()/255
def ce(z,y):
    return z.logsumexp(-1)-z.gather(-1,y[...,None]).squeeze(-1)
def wrong(z,q,y):
    mask=F.one_hot(y,100).bool()
    q=q.masked_fill(mask,0);q=q/q.sum(-1,keepdim=True).clamp_min(1e-14)
    p=(z/2).masked_fill(mask,-1e4).log_softmax(-1)
    return F.kl_div(p,q,reduction="none").sum(-1)
def kl(z,q):
    return F.kl_div((z/2).log_softmax(-1),q,reduction="none").sum(-1)
def moments(z,y):
    losses=ce(z,y)
    return torch.stack([losses.mean(1),losses.std(1,unbiased=False)],1).flatten()

def run():
    cli=argparse.ArgumentParser()
    cli.add_argument("--overcorrect",action="store_true")
    late=cli.parse_args().overcorrect
    start=time.monotonic();torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    assert sha(OUT/"A32.pt")=="96716abb20837c57c7e692150acd5d85a73e70f295c2ccaab87ad7759a79c7a2"
    c=state(ROOT/"historical/record4112.pt");a=state(OUT/"A32.pt")
    A=model(a);C=model(c);O=model(state(ROOT/"model.pt"))
    pixels=tt.load_forget_images()
    native=state(Path("./work/cnsu_original/never64.pt")); pg=state(ROOT/"historical/pg.pt")
    y=native["y"].cuda().long();assert torch.equal(y,pg["y"].cuda())
    q=native["p2"].cuda().float();q=q/q.sum(1,keepdim=True)
    kimi=pg["pg"].cuda().float();kimi=kimi/kimi.sum(1,keepdim=True)
    rr=torch.Generator().manual_seed(79623)
    ifit=torch.randperm(2500,generator=rr)[:1600]
    itest=torch.arange(2500,3000)
    syn=np.load(ROOT/"gen/gen_test.npz"); fresh=np.load(FRESH/"audited_images.npz")
    sx=torch.from_numpy(syn["images"][:650]); sy=torch.from_numpy(syn["cls"][:650])
    fx=torch.from_numpy(fresh["images"]);fy=torch.from_numpy(fresh["cls"])
    def scan(px,ids,aug_seed=None,need_orig=False):
        gen=torch.Generator(device="cuda").manual_seed(aug_seed or 1)
        ah=[]; cz=[];orig=[]
        hook=A.head.register_forward_pre_hook(lambda m,args:ah.append(args[0].float().detach()))
        with torch.no_grad():
            for ix in ids.split(32):
                xx=prep(px[ix])
                if aug_seed:xx=alter(xx,gen,.7)
                xx=2*xx-1
                with torch.autocast("cuda",dtype=torch.float16):
                    A(xx);cz.append(C(xx).float())
                    if need_orig:orig.append(O(xx).float())
        hook.remove()
        hh=torch.cat(ah)
        return torch.cat([hh,torch.ones(len(hh),1,device="cuda")],1),torch.cat(cz),torch.cat(orig) if orig else None
    f0=scan(pixels,ifit);f1=scan(pixels,ifit,64201)
    d0=scan(pixels,itest);d1=scan(pixels,itest,64202,True)
    G,_,_=scan(sx,torch.arange(256))
    H,zc_syn,zo_syn=scan(sx,torch.arange(384,650),need_orig=True)
    J,zc_fresh,zo_fresh=scan(fx,torch.arange(len(fx)),need_orig=True)
    print("feature cache ready",time.monotonic()-start,flush=True)
    t=torch.stack([f0[0],f1[0]])
    h=torch.stack([d0[0],d1[0]])
    Y=y[ifit].expand(2,-1);yd=y[itest].expand(2,-1)
    zcfit=torch.stack([f0[1],f1[1]]);zcdev=torch.stack([d0[1],d1[1]])
    qfit=q[itest]; choose=ce(d1[2],yd[1])>ce(2*qfit.clamp_min(1e-12).log(),yd[1])
    Q=torch.stack([qfit,torch.where(choose[:,None],(d1[2]/2).softmax(1),qfit)])
    K=kimi[itest].expand(2,-1,-1)
    w=stack_head(a)
    base=moments(t@w,Y); want=moments(zcfit,Y)
    regular=t.flatten(0,1);cov=regular.T@regular/len(regular)+8*G.T@G/len(G)
    cov.diagonal().add_(.05)
    def solve(w,targ):
        grads=[]
        for v in t:
            zv=v@w;yv=Y[0];loss=ce(zv,yv)
            dz=(zv.softmax(1)-F.one_hot(yv,100).float())/len(v)
            grads.extend([v.T@dz,v.T@(dz*((loss-loss.mean())/loss.std(unbiased=False).clamp_min(1e-8))[:,None])])
        gr=torch.stack(grads);r,d,k=gr.shape
        inv=torch.linalg.solve(cov,gr.permute(1,0,2).reshape(d,-1)).reshape(d,r,k).permute(1,0,2)
        ker=gr.flatten(1)@inv.flatten(1).T;ker.diagonal().add_(2e-6)
        return torch.einsum("i,ijk->jk",torch.linalg.solve(ker,targ-moments(t@w,Y)),inv)
    def synth(z,zo,yy):
        yy=yy.cuda();mask=yy>=0
        return {"kl":float(kl(z,(zo/2).softmax(1)).mean()),"agreement":float((z.argmax(1)==zo.argmax(1)).float().mean()),
                "correct":int((z[mask].argmax(1)==yy[mask]).sum()),"labelled":int(mask.sum())}
    def metrics(z,zs,zf):
        return {"moments":moments(z,yd).tolist(),"global_p2":kl(z,Q).mean(1).tolist(),"global_kimi":kl(z,K).mean(1).tolist(),
                "conditional_p2":wrong(z,Q,yd).mean(1).tolist(),"conditional_kimi":wrong(z,K,yd).mean(1).tolist(),
                "synth266":synth(zs,zo_syn,sy[384:650]),"synth495":synth(zf,zo_fresh,fy)}
    report={"anchor_sha256":sha(OUT/"A32.pt"),"n_fit":len(ifit),"n_dev":len(itest),"train_seed":79623,"fit_aug_seed":64201,
            "dev_aug_seed":64202,"synth_for_fit":"gen_test:0:256","synth_for_dev":["gen_test:384:650","fresh 495 seed681734119"],
            "fit_initial":base.tolist(),"fit_desired":want.tolist(),"rows":{}}
    rows=report["rows"]
    rows["champion"]=metrics(zcdev,zc_syn,zc_fresh)
    rows["a32"]=metrics(h@w,H@w,J@w)
    tests=(("repair160",1.60),("repair185",1.85)) if late else (("repair60",.6),("repair100",1.))
    rng=np.random.default_rng(40319)
    draws=rng.integers(0,len(itest),size=(2500,len(itest)))
    for tag,alpha in tests:
        W=w.clone();target=base+alpha*(want-base)
        for i in range(5):W+=solve(W,target)
        row=metrics(h@W,H@W,J@W);row["fit_moments"]=moments(t@W,Y).tolist()
        row["head_delta_norm"]=float((W-w).norm())
        znew=h@W
        cis={}
        for name,stat,qref in (("global_p2",kl,Q),("global_kimi",kl,K),("conditional_p2",wrong,Q),("conditional_kimi",wrong,K)):
            current=stat(znew,qref,yd) if stat==wrong else stat(znew,qref)
            anchor=stat(zcdev,qref,yd) if stat==wrong else stat(zcdev,qref)
            paired=(current-anchor).cpu().numpy()
            bounds=[np.quantile(v[draws].mean(1),[.025,.975]).tolist() for v in paired]
            cis[name]=[[float(v.mean()),*bound] for v,bound in zip(paired,bounds)]
        row["paired_KL_change_ci95"]=cis
        ck=dict(a);ck["head.weight"]=W[:-1].T.detach().contiguous().cpu();ck["head.bias"]=W[-1].detach().contiguous().cpu()
        assert len(ck)==176 and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in ck.values())
        file=OUT/("cnsu_a32_"+tag+".pt");torch.save(ck,file)
        loaded=state(file);A.load_state_dict(loaded,strict=True)
        assert all(torch.equal(v,a[k]) for k,v in loaded.items() if k not in ("head.weight","head.bias"))
        row["sha256"]=sha(file);rows[tag]=row
    cm=np.asarray(rows["champion"]["moments"]);am=np.asarray(rows["a32"]["moments"])
    for label,row in rows.items():
        row["moment_max_drift"]=float(np.max(np.abs(np.array(row["moments"])-cm)))
        row["moment_recovery_dev"]=float(1-np.linalg.norm(np.array(row["moments"])-cm)/max(np.linalg.norm(am-cm),1e-10))
        row["fraction_initial_global_p2_gain"]=((np.array(rows["champion"]["global_p2"])-np.array(row["global_p2"]))/(np.array(rows["champion"]["global_p2"])-np.array(rows["a32"]["global_p2"]))).tolist()
    eligible=[n for n in (("repair60","repair100") if not late else ()) if min(rows[n]["fraction_initial_global_p2_gain"])>.15 and rows[n]["moment_recovery_dev"]>0 and rows[n]["synth266"]["correct"]>=rows["champion"]["synth266"]["correct"]-1 and rows[n]["synth495"]["correct"]>=rows["champion"]["synth495"]["correct"]-1]
    frozen=min(eligible,key=lambda n:rows[n]["moment_max_drift"]) if eligible else None
    if late:
        options=[]
        backup_options=[]
        for tag,_ in tests:
            rr=rows[tag]
            bounds=[c for both in rr["paired_KL_change_ci95"].values() for c in both]
            significant_inverse=any(lo>0 for val,lo,hi in bounds)
            all_directions_positive=all(val<=0 for val,lo,hi in bounds)
            retention=all(rr[b]["correct"]>=rows["champion"][b]["correct"]-1 for b in ("synth266","synth495"))
            rr["significant_KL_inversion"]=significant_inverse
            rr["all_KL_point_estimates_improve"]=all_directions_positive
            rr["retention_guard"]=retention
            if not significant_inverse:
                backup_options.append((not all_directions_positive,rr["moment_max_drift"],tag))
            if not significant_inverse and retention:
                options.append((not all_directions_positive,rr["moment_max_drift"],tag))
        frozen=min(options)[2] if options else None
        backup=min(backup_options)[2] if backup_options else None
        report["backup_selection_with_measured_retention_risk"]=backup
        report["backup_sha256"]=rows[backup]["sha256"] if backup else None
    report["development_selection"]=frozen
    report["elapsed_s"]=time.monotonic()-start;report["GPU_GB"]=torch.cuda.max_memory_allocated()/1e9
    (OUT/("repair_report_ext.json" if late else "repair_report.json")).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    run()
