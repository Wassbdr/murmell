"""Transport the true-label margin change while holding the other classes' odds near the anchor."""
import json
import hashlib
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

home=Path(os.environ.get("T3_HOME", "."))
out=Path(os.environ.get("T3_OUT", "out"))
sys.path.insert(0,str(home))
os.chdir(home)
import task_template as tt
from t3_retain import load_gen,logits
from t3_consist import alter
tt.DEVICE="cpu"
torch.set_num_threads(2)
transplant="--restaurer-champion" in sys.argv
anchor_path=out/"codex_thermal_mlp_pure100.pt" if transplant else home/"out/kimi_conditional_soup50.pt"
anchor=torch.load(anchor_path,map_location="cpu",weights_only=True,mmap=True)
contrast=torch.load(out/("codex_marge_isolee.pt" if transplant else "codex_tete_contraste.pt"),map_location="cpu",weights_only=True,mmap=True)
orig=tt.load_model().cuda().eval()
model=tt.build_model().cuda().eval()
model.load_state_dict(anchor,strict=True)
xs=((tt.preprocess(tt.load_forget_images())+1)/2).contiguous()
y=logits(orig,xs).argmax(1).cuda()
gc, c=load_gen("test:650:852")
gfit,_=load_gen("test:0:650")
with torch.no_grad():
    original_g=logits(orig,gc).cuda()
del orig
@torch.inference_mode()
def latent(X,strength=0,seed=0):
    pieces=[]
    def hook(module,args):
        pieces.append(args[0].detach().float())
    handle=model.head.register_forward_pre_hook(hook)
    rng=torch.Generator(device="cuda").manual_seed(seed)
    try:
        for start in range(0,len(X),75):
            px=X[start:start+75].cuda()
            if strength: px=alter(px,rng,strength)
            with torch.autocast("cuda",dtype=torch.float16):
                model(px*2-1)
    finally:
        handle.remove()
    result=torch.cat(pieces)
    return torch.cat([result,torch.ones(len(result),1,device="cuda")],1)
V=torch.stack([latent(xs),latent(xs,.5,212),latent(xs,1.,771)])
Gc,Gfit=latent(gc),latent(gfit)
w=lambda st:torch.cat((st["head.weight"].T,st["head.bias"][None]),0).cuda()
a,b=w(anchor),w(contrast)
z=V@a
truth=y.expand(len(V),-1)
def ce(Z,yy):
    return torch.logsumexp(Z,-1)-Z.gather(-1,yy[...,None]).squeeze(-1)
def logexpm1(x):
    x=x.clamp_min(1e-5)
    return x+torch.log(-torch.expm1(-x))
def observed(state):
    model.load_state_dict(state,strict=True)
    observed_z=[]
    for amp,seed in ((0,0),(.5,212),(1.,771)):
        generator=torch.Generator(device="cuda").manual_seed(seed)
        transform=(lambda v,r: alter(v,r,amp)) if amp else None
        observed_z.append(logits(model,xs,bs=75,f=transform,g=generator).cuda())
    return torch.stack(observed_z)

target_logits=V@b
if transplant:
    target_logits=observed(contrast)
    old_state=torch.load(home/"out/kimi_conditional_soup50.pt",map_location="cpu",weights_only=True,mmap=True)
    old_logits=observed(old_state)
    model.load_state_dict(anchor,strict=True)
    amplitude=float(sys.argv[sys.argv.index("--restaurer-champion")+1])
    wanted_ce=(ce(old_logits,truth)+amplitude*(ce(target_logits,truth)-ce(old_logits,truth))).clamp_min(.00003)
else:
    wanted_ce=ce(target_logits,truth)
want=(logexpm1(ce(z,truth))-logexpm1(wanted_ce)).clamp(-1.5,1.5)
ids=torch.randperm(len(xs),generator=torch.Generator().manual_seed(104357))
tr,va,te=ids[:2200],ids[2200:2750],ids[2750:]
T=V[:,tr].reshape(-1,V.shape[-1])
target=want[:,tr].flatten()
Y=truth[:,tr].flatten()
def kl(P,Q):
    return F.kl_div((Q/2).log_softmax(-1),(P/2).softmax(-1),reduction="none").sum(-1)
def check(D,ids):
    sub=z[:,ids]; lab=truth[:,ids]
    target_z=target_logits[:,ids]
    target_ce=wanted_ce[:,ids]
    final=sub+V[:,ids]@D
    null0=(ce(sub,lab)-target_ce).square().mean()
    mse=(ce(final,lab)-target_ce).square().mean()
    corr=1.-float(mse/null0)
    mask=F.one_hot(lab,100).bool()
    wrong=lambda l:l.masked_fill(mask,-10000)
    alt=float(kl(wrong(sub),wrong(final)).mean())
    raw=float(kl(wrong(sub),wrong(target_z)).mean())
    return dict(explained_ce_mse=corr,alternate_kl=alt,contrast_alternate_kl=raw,
                ce_by_view=[[float(v.mean()),float(v.std())] for v in ce(final,lab)],
                original_ce_by_view=[[float(v.mean()),float(v.std())] for v in ce(sub,lab)])
def fit(weight):
    gram=weight*(T.T@T)+.3*(Gfit.T@Gfit)+torch.eye(T.shape[1],device="cuda")*24
    columns=[]
    for k in range(100):
        X=T[Y==k]
        q=target[Y==k]
        columns.append(torch.linalg.solve(gram+X.T@X,X.T@q))
    return torch.stack(columns,1)
def retain(D):
    g=Gc@(a+D)
    valid=c.cuda()>=0
    return dict(kl=float(kl(original_g,g).mean()),accuracy=float((g.argmax(1)[valid]==c.cuda()[valid]).float().mean()))
runs=[]
for weight in (.004,.016,.06):
    D=fit(weight)
    val=check(D,va)
    # Smaller wrong-class movement is better at equal held-out margin fit.
    utility=val["explained_ce_mse"]-max(0.,val["alternate_kl"]/val["contrast_alternate_kl"]-.4)
    runs.append((utility,D,val,weight))
utility,D,val,weight=max(runs,key=lambda item:item[0])
test=check(D,te)
record=dict(n_train=len(tr),n_val=len(va),n_test=len(te),weight=weight,validation=val,confirmed=test,
            base_retain=retain(torch.zeros_like(D)),retain=retain(D),variants=[v[2] for v in runs])
prefix=("codex_greffe_marge_"+str(round(amplitude*100))) if transplant else "codex_marge_isolee"
if test["explained_ce_mse"]>0 and (test["alternate_kl"]<.8*test["contrast_alternate_kl"] or (transplant and test["alternate_kl"]<.0025)):
    state=dict(anchor)
    state["head.weight"]=(a+D)[:-1].T.contiguous().cpu()
    state["head.bias"]=(a+D)[-1].contiguous().cpu()
    model.load_state_dict(state,strict=True)
    p=out/(prefix+".pt")
    torch.save(state,p)
    with p.open("rb") as f:record["sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
(out/((prefix+".json") if transplant else "marge_isolee.json")).write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
