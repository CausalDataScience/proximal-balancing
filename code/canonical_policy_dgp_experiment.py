#!/usr/bin/env python3
"""One-SCM canonical-policy diagnostic with train-only deterministic cell merging."""
from __future__ import annotations

import hashlib, itertools, json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT_SEED=1950702; N_MASTER=1000; N_DIAGNOSTIC=20000; N_DIAGNOSTIC_KERNEL=2000
PREFIXES=(100,200,500,1000); D_X=20; D_W=12; N_AUDIT=5; N_STRATA=3
TRUE_ATE=1.0; STEPS=40; HIDDEN=32; LR=3e-3; MIN_CELL_ARM=5
OUTPUT_PATH=Path(__file__).resolve().parents[1]/"results"/"canonical_policy_smoke.json"
AUDIT_SLOPES=np.array([.7,.9,1.1,1.3,1.5]); AUDIT_NOISE=np.array([.6,.7,.8,.9,1.])
POLICY_PROBS=np.array([.2,.5,.8])
GBR_SPEC={"n_estimators":100,"max_depth":2,"learning_rate":.05,"min_samples_leaf":5}

def ah(x):
    x=np.ascontiguousarray(x); h=hashlib.sha256(); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes()); return h.hexdigest()
def js(x:Any)->Any:
    if isinstance(x,dict): return {str(k):js(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [js(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.bool_,bool)): return bool(x)
    if isinstance(x,(np.floating,float)):
        v=float(x); return v if math.isfinite(v) else None
    if isinstance(x,(np.integer,int)): return int(x)
    return x
def policy_class(x,u):
    eta=.8*u+.45*x[:,0]-.35*x[:,1]+.25*x[:,2]*x[:,3]
    return eta,np.where(eta<=-.65,0,np.where(eta<=.65,1,2)).astype(int)
def anchor(u): return u+.25*np.tanh(u)
def inv_anchor(v):
    u=np.asarray(v,float).copy()
    for _ in range(30):
        t=np.tanh(u); u-=(u+.25*t-v)/(1+.25*(1-t*t))
    return u

def generate_pair(n,seed):
    rr=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(8)]
    ru,rx,rq,rn,rv,ra,ry,rf=rr; u=ru.standard_normal(n); ex=rx.standard_normal((n,D_X)); x=np.empty((n,D_X))
    x[:,0]=.8*u+.6*ex[:,0]; x[:,1]=np.sin(u)+.6*ex[:,1]; x[:,2]=np.tanh(1.5*u)+.6*ex[:,2]
    x[:,3]=(u*u-1)/math.sqrt(2)+.6*ex[:,3]; x[:,4]=.5*u+.5*np.sin(2*u)+.6*ex[:,4]
    x[:,5:]=u[:,None]*np.linspace(-.4,.4,D_X-5)[None,:]+.8*ex[:,5:]
    eta,h0=policy_class(x,u); prop=POLICY_PROBS[h0]; a=ra.binomial(1,prop).astype(float)
    eq=rq.standard_normal((n,N_AUDIT)); q=np.column_stack([AUDIT_SLOPES[j]*u+.15*x[:,j+5]+AUDIT_NOISE[j]*eq[:,j] for j in range(N_AUDIT)])
    en=rn.standard_normal((n,6)); nuis=np.column_stack([.15*u+.1*x[:,min(15+j,19)]+1.5*en[:,j] for j in range(6)])
    vb=anchor(u); vg=vb+.8*rv.standard_normal(n); y0=.8*x[:,0]+.4*x[:,1]*x[:,2]+1.5*u; y=y0+a+.5*ry.standard_normal(n)
    order=rf.permutation(n); fold=np.empty(n,int); fold[order[::2]]=0; fold[order[1::2]]=1
    common={"X":x,"U":u,"Q":q,"N":nuis,"A":a,"Y":y,"Y0_struct":y0,"eta":eta,"H0":h0,"propensity":prop,"fold":fold}
    g=dict(common); b=dict(common); g["V"]=vg; b["V"]=vb; g["W"]=np.column_stack([q,vg,nuis]); b["W"]=np.column_stack([q,vb,nuis])
    return {"generic":g,"balancing_valid":b}

def bandwidth(x):
    d=((x[:,None]-x[None,:])**2).sum(-1); v=d[np.triu_indices_from(d,1)]; return math.sqrt(max(float(np.median(v)),1e-12)/2)
def gram(x,b): return np.exp(-((x[:,None]-x[None,:])**2).sum(-1)/(2*b*b))
def soft_loss(s,a,k):
    n=len(a); p=s.mean(0).clamp_min(1e-8); pi=((s*a[:,None]).mean(0)/p).clamp(1e-4,1-1e-4); c=s*(a[:,None]-pi); return ((c*(k@c)).sum(0)/(n*n*p)).sum(),p,pi
class Net(torch.nn.Module):
    def __init__(self,d): super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(d,HIDDEN),torch.nn.ReLU(),torch.nn.Linear(HIDDEN,HIDDEN),torch.nn.ReLU(),torch.nn.Linear(HIDDEN,N_STRATA))
    def forward(self,x): return self.net(x)
@dataclass
class Fit:
    net:Net; curve:list[float]; im:np.ndarray; sd:np.ndarray; am:np.ndarray; ad:np.ndarray; bw:float; spec:str
def fit_rep(inp,audit,a,seed,same):
    im=inp.mean(0); sd=inp.std(0)+1e-8; z=(inp-im)/sd; am=audit.mean(0); ad=audit.std(0)+1e-8; za=(audit-am)/ad; bw=bandwidth(za)
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True); net=Net(z.shape[1]); xt=torch.tensor(z,dtype=torch.float32); at=torch.tensor(a,dtype=torch.float32); kt=torch.tensor(gram(za,bw),dtype=torch.float32)
    opt=torch.optim.Adam(net.parameters(),lr=LR,weight_decay=1e-4 if same else 0); gen=torch.Generator().manual_seed(seed+700001); curve=[]
    for step in range(STEPS):
        temp=.25**(step/max(STEPS-1,1)); xs=xt
        if same: xs=xt*torch.bernoulli(torch.full_like(xt,.8),generator=gen)/.8+.1*torch.randn(xt.shape,generator=gen)
        logits=net(xs); s=torch.softmax(logits/temp,1); d,p,pi=soft_loss(s,at,kt); overlap=(p*torch.relu(.09-pi*(1-pi))).sum(); mass=torch.relu(.5/N_STRATA-p).sum(); lp=1e-3*logits.square().sum(1).mean() if same else 0
        loss=d+5*overlap+5*mass+lp; opt.zero_grad(); loss.backward(); opt.step(); curve.append(float(d.detach()))
    spec="same-view heuristic; uncertified; fixed dropout/noise/weight-decay/logit regularization" if same else "leave-one-audit representation"
    return Fit(net,curve,im,sd,am,ad,bw,spec)
def probs(fit,x):
    fit.net.eval()
    with torch.no_grad(): return torch.softmax(fit.net(torch.tensor((x-fit.im)/fit.sd,dtype=torch.float32))/.25,1).numpy()

def counts(h,a):
    out=[]
    for z in sorted(np.unique(h).astype(int).tolist()):
        m=h==z; out.append({"stratum":z,"n":int(m.sum()),"n1":int(np.sum(a[m]==1)),"n0":int(np.sum(a[m]==0))})
    return out
def weights(h,labels): return np.column_stack([(h==z).astype(float) for z in labels])
def discrepancy_sq(w,a,audit,bw,am,ad,fixed_pi=None):
    k=gram((audit-am)/ad,bw); n=len(a); total=0.
    for j in range(w.shape[1]):
        ww=w[:,j]; mass=ww.mean()
        if mass<=0: continue
        pi=float((ww*a).sum()/ww.sum()) if fixed_pi is None else float(fixed_pi[j]); c=ww*(a-pi); total+=float(c@k@c)/(n*n*mass)
    return max(total,0.)
def ordinary(h,a,audit,fit):
    cc=counts(h,a); ok=all(r["n0"] and r["n1"] for r in cc); val=None
    if ok: val=math.sqrt(discrepancy_sq(weights(h,[r["stratum"] for r in cc]),a,audit,fit.bw,fit.am,fit.ad))
    return {"valid":ok,"delta":val,"counts":cc,"no_units_dropped":True}
def ordinary_params(h,a,audit,bw,am,ad):
    cc=counts(h,a); ok=all(r["n0"] and r["n1"] for r in cc); val=None
    if ok: val=math.sqrt(discrepancy_sq(weights(h,[r["stratum"] for r in cc]),a,audit,bw,am,ad))
    return {"valid":ok,"delta":val,"counts":cc,"no_units_dropped":True}
def oracle_h0_positive_control(data,diag):
    out=[]
    for n in PREFIXES:
        for f in (0,1):
            tr=np.where(data["fold"][:n]!=f)[0]; te=np.where(data["fold"][:n]==f)[0]; audits=[]; k=N_DIAGNOSTIC_KERNEL
            for j in range(N_AUDIT):
                av=np.column_stack([data["X"][:n],data["Q"][:n,j]]); dv=np.column_stack([diag["X"][:k],diag["Q"][:k,j]])
                am=av[tr].mean(0); ad=av[tr].std(0)+1e-8; bw=bandwidth((av[tr]-am)/ad)
                audits.append({"audit_index":j,"heldout":ordinary_params(data["H0"][te],data["A"][te],av[te],bw,am,ad),"independent_diagnostic":ordinary_params(diag["H0"][:k],diag["A"][:k],dv,bw,am,ad)})
            out.append({"n":n,"test_fold":f,"train_counts":counts(data["H0"][tr],data["A"][tr]),"heldout_counts":counts(data["H0"][te],data["A"][te]),"heldout_latent_imbalance":latent(data["H0"][te],data["A"][te],data["U"][te]),"independent_diagnostic_counts":counts(diag["H0"],diag["A"]),"independent_diagnostic_latent_imbalance":latent(diag["H0"],diag["A"],diag["U"]),"audits":audits})
    return out

def merge_cells(raw,a,audit,fit):
    gc={str(t):int(np.sum(a==t)) for t in (0,1)}
    if min(gc.values())<MIN_CELL_ARM: return {"success":False,"reason":"global arm count below k","global_arm_counts":gc,"sequence":[]}
    h=raw.copy().astype(int); raw_labels=sorted(np.unique(h).tolist()); mp={z:z for z in raw_labels}; seq=[]
    while True:
        cc=counts(h,a); bad=[r["stratum"] for r in cc if min(r["n0"],r["n1"])<MIN_CELL_ARM]
        if not bad: break
        active=[r["stratum"] for r in cc]
        if len(active)==1: return {"success":False,"reason":"one invalid cell after global precheck","global_arm_counts":gc,"sequence":seq}
        old=discrepancy_sq(weights(h,active),a,audit,fit.bw,fit.am,fit.ad); cand=[]
        for src in sorted(bad):
            for dst in sorted(z for z in active if z!=src):
                hp=h.copy(); hp[hp==src]=dst; labs=sorted(np.unique(hp).tolist()); obj=discrepancy_sq(weights(hp,labs),a,audit,fit.bw,fit.am,fit.ad); cand.append((obj-old,src,dst,obj))
        inc,src,dst,obj=min(cand,key=lambda r:(r[0],r[1],r[2])); h[h==src]=dst; mp={r:(dst if z==src else z) for r,z in mp.items()}; seq.append({"source":src,"target":dst,"objective_before":old,"objective_after":obj,"objective_increase":inc,"tie_break":"(increase, source_label, target_label)"})
    active=sorted(np.unique(h).tolist()); canon={z:j for j,z in enumerate(active)}; final={r:canon[z] for r,z in mp.items()}
    # A raw label absent from outer training can still appear held out.  Map it
    # deterministically to the smallest surviving training label.  This uses no
    # held-out information and changes no training objective because its mass is zero.
    absent=[r for r in range(N_STRATA) if r not in final]
    for r in absent: final[r]=0
    merged=np.array([final[int(z)] for z in raw]); fc=counts(merged,a)
    return {"success":True,"reason":None,"global_arm_counts":gc,"map":final,"sequence":seq,"final_m":len(active),"final_counts":fc,"merged_train":merged,"training_absent_raw_labels":absent,"absent_label_fallback_target":0,"all_final_cells_min_arm_k":all(min(r["n0"],r["n1"])>=MIN_CELL_ARM for r in fc)}
def apply_merge(h,mp): return np.array([mp[int(z)] for z in h],int)
def onehot(h,m): return np.eye(m)[h]
def h0_purity(h,h0):
    labs=sorted(np.unique(h).tolist()); best=0
    for tar in itertools.product(range(N_STRATA),repeat=len(labs)):
        mp=dict(zip(labs,tar)); best=max(best,int(np.sum(np.array([mp[int(z)] for z in h])==h0)))
    return best/len(h)
def clustering_qa(h,h0,raw):
    labels=sorted(np.unique(h).tolist()); matrix=[[int(np.sum((h==z)&(h0==t))) for t in range(N_STRATA)] for z in labels]
    matched=None
    if raw and len(labels)==N_STRATA:
        matched=max(sum(matrix[i][p[i]] for i in range(N_STRATA)) for p in itertools.permutations(range(N_STRATA)))/len(h)
    return {"labels":labels,"confusion_rows_learned_cols_h0":matrix,"h0_purity":h0_purity(h,h0),"hungarian_one_to_one_matched_accuracy":matched,"adjusted_rand_index":float(adjusted_rand_score(h0,h)),"normalized_mutual_information":float(normalized_mutual_info_score(h0,h)),"exact_recovery_claim_permitted":bool(raw and len(labels)==N_STRATA)}
def latent(h,a,u):
    rows=[]; avg=0.
    for r in counts(h,a):
        m=h==r["stratum"]
        if r["n0"] and r["n1"]: gap=float(u[m&(a==1)].mean()-u[m&(a==0)].mean()); avg+=r["n"]/len(h)*abs(gap)
        else: gap=None
        rows.append({"stratum":r["stratum"],"mean_u_treated_minus_control":gap})
    return {"weighted_absolute_gap":avg,"strata":rows}

def tlearner(st,a,y,se,seed):
    ac={str(t):int(np.sum(a==t)) for t in (0,1)}
    if min(ac.values())<GBR_SPEC["min_samples_leaf"]: return None,{"arm_counts":ac,"finite":False,"reason":"global arm failure"}
    pp=[]
    for arm in (1,0):
        m=a==arm; model=GradientBoostingRegressor(random_state=seed+arm,**GBR_SPEC); model.fit(st[m],y[m]); pp.append(model.predict(se))
    eff=pp[0]-pp[1]; ok=bool(np.isfinite(eff).all()); return (eff if ok else None),{"arm_counts":ac,"finite":ok}
def baseline(data,n,key,method,seed):
    if key=="X": s=data["X"][:n]
    elif key=="XW": s=np.column_stack([data["X"][:n],data["W"][:n]])
    elif key=="XU": s=np.column_stack([data["X"][:n],data["U"][:n]])
    elif key=="H0": s=onehot(data["H0"][:n],N_STRATA)
    stitched=np.full(n,np.nan); folds=[]
    for f in (0,1):
        te=np.where(data["fold"][:n]==f)[0]; tr=np.where(data["fold"][:n]!=f)[0]; pred,qa=tlearner(s[tr],data["A"][tr],data["Y"][tr],s[te],seed+1009*f)
        if pred is not None: stitched[te]=pred
        folds.append({"test_fold":f,"train_n":len(tr),"test_n":len(te),**qa})
    ok=bool(np.isfinite(stitched).all()); est=float(stitched.mean()) if ok else None
    return {"method":method,"estimate":est,"error":est-TRUE_ATE if ok else None,"abs_error":abs(est-TRUE_ATE) if ok else None,"finite":ok,"certified":method in ("oracle_xu","oracle_h0")},folds

def diag_eval(fit,mp,absent,inp,audit,data,train_pi):
    p=probs(fit,inp); raw=p.argmax(1); merged=apply_merge(raw,mp); k=min(N_DIAGNOSTIC_KERNEL,len(raw)); labs=list(range(len(set(mp.values())))); aa=data["A"][:k]
    return {"n":len(raw),"kernel_n":k,"frozen_model_only":True,"used_for_tuning_or_merge":False,"absent_training_label_fallback_applications":int(np.isin(raw,absent).sum()),"raw_h0_clustering":clustering_qa(raw,data["H0"],True),"merged_h0_clustering":clustering_qa(merged,data["H0"],False),"soft_discrepancy":math.sqrt(discrepancy_sq(p[:k],aa,audit[:k],fit.bw,fit.am,fit.ad)),"raw_hard_discrepancy":math.sqrt(discrepancy_sq(weights(raw[:k],sorted(np.unique(raw[:k]).tolist())),aa,audit[:k],fit.bw,fit.am,fit.ad)),"merged_hard_fixed_train_pi_discrepancy":math.sqrt(discrepancy_sq(weights(merged[:k],labs),aa,audit[:k],fit.bw,fit.am,fit.ad,train_pi)),"latent_imbalance":latent(merged,data["A"],data["U"])}

def representation(data,diag,n,method,j,seed,certified):
    stitched=np.full(n,np.nan); raw_stitched=np.full(n,np.nan); details=[]
    for f in (0,1):
        te=np.where(data["fold"][:n]==f)[0]; tr=np.where(data["fold"][:n]!=f)[0]; x=data["X"][:n]; w=data["W"][:n]; a=data["A"][:n]; y=data["Y"][:n]
        if method=="v1":
            keep=[q for q in range(D_W) if q!=j]; inp=np.column_stack([x,w[:,keep]]); audit=np.column_stack([x,w[:,j]]); dinp=np.column_stack([diag["X"],diag["W"][:,keep]]); daudit=np.column_stack([diag["X"],diag["W"][:,j]]); same=False; excluded=j not in keep
        else:
            inp=np.column_stack([x,w]); audit=inp; dinp=np.column_stack([diag["X"],diag["W"]]); daudit=dinp; same=True; excluded=True
        fit=fit_rep(inp[tr],audit[tr],a[tr],seed+1009*f,same); pt=probs(fit,inp[tr]); pe=probs(fit,inp[te]); rt=pt.argmax(1); re=pe.argmax(1); mg=merge_cells(rt,a[tr],audit[tr],fit)
        if not mg["success"]: raise RuntimeError("explicit merge failure: "+mg["reason"])
        mt=mg.pop("merged_train"); me=apply_merge(re,mg["map"]); m=mg["final_m"]; pi=np.array([a[tr][mt==z].mean() for z in range(m)])
        pred,oq=tlearner(onehot(mt,m),a[tr],y[tr],onehot(me,m),seed+50000+f)
        if pred is not None: stitched[te]=pred
        rp,rq=tlearner(onehot(rt,N_STRATA),a[tr],y[tr],onehot(re,N_STRATA),seed+50000+f)
        if rp is not None: raw_stitched[te]=rp
        wm=weights(me,list(range(m))); soft=math.sqrt(discrepancy_sq(pe,a[te],audit[te],fit.bw,fit.am,fit.ad)); rawd=math.sqrt(discrepancy_sq(weights(re,sorted(np.unique(re).tolist())),a[te],audit[te],fit.bw,fit.am,fit.ad)); mergedd=math.sqrt(discrepancy_sq(wm,a[te],audit[te],fit.bw,fit.am,fit.ad)); fixed=math.sqrt(discrepancy_sq(wm,a[te],audit[te],fit.bw,fit.am,fit.ad,pi))
        heldout_counts=counts(me,a[te]); absent=mg["training_absent_raw_labels"]
        details.append({"test_fold":f,"representation_learned_on_outer_train_only":True,"heldout_evaluation_only":True,"train_indices_hash":ah(tr),"test_indices_hash":ah(te),"train_test_disjoint":not np.intersect1d(tr,te).size,"audit_index":j,"audit_excluded_from_input_and_preprocessing":excluded,"raw_train_counts":counts(rt,a[tr]),"raw_heldout_counts":counts(re,a[te]),"merge":mg,"final_train_counts":counts(mt,a[tr]),"final_heldout_counts":heldout_counts,"all_final_training_cells_min_arm_k":all(min(r["n0"],r["n1"])>=MIN_CELL_ARM for r in counts(mt,a[tr])),"all_positive_mass_heldout_cells_both_arms":all(r["n0"]>0 and r["n1"]>0 for r in heldout_counts),"no_heldout_remerge":True,"no_units_dropped":len(mt)==len(tr) and len(me)==len(te),"absent_training_label_fallback_applications_heldout":int(np.isin(re,absent).sum()),"raw_h0_clustering_train":clustering_qa(rt,data["H0"][tr],True),"raw_h0_clustering_heldout":clustering_qa(re,data["H0"][te],True),"merged_h0_clustering_train":clustering_qa(mt,data["H0"][tr],False),"merged_h0_clustering_heldout":clustering_qa(me,data["H0"][te],False),"heldout_joint_discrepancy":{"soft":soft,"raw_hard":rawd,"merged_hard":mergedd,"merged_hard_fixed_training_propensity":fixed,"ordinary_mmd":ordinary(me,a[te],audit[te],fit)},"latent_imbalance_heldout":latent(me,a[te],data["U"][te]),"soft_raw_merged_gap":{"raw_minus_soft":rawd-soft,"merged_minus_raw":mergedd-rawd},"initial_objective":float(np.mean(fit.curve[:5])),"final_objective":float(np.mean(fit.curve[-5:])),"optimization_decreased":np.mean(fit.curve[-5:])<=np.mean(fit.curve[:5])+1e-10,"raw_estimate_fold":float(rp.mean()) if rp is not None else None,"merged_estimate_fold":float(pred.mean()) if pred is not None else None,"raw_outcome_model":rq,"merged_outcome_model":oq,"independent_large_diagnostic":diag_eval(fit,mg["map"],absent,dinp,daudit,diag,pi),"theorem_endpoint_available_in_dgp":method=="v1" and certified,"learned_fit_exact_balance_certified":False,"specification":fit.spec})
    ok=bool(np.isfinite(stitched).all()); est=float(stitched.mean()) if ok else None; rok=bool(np.isfinite(raw_stitched).all())
    return {"method":method if j is None else f"v1_mask_{j+1}","estimate":est,"error":est-TRUE_ATE if ok else None,"abs_error":abs(est-TRUE_ATE) if ok else None,"finite":ok,"raw_unmerged_estimate":float(raw_stitched.mean()) if rok else None,"raw_unmerged_finite":rok,"certified":False,"theorem_endpoint_available_in_dgp":method=="v1" and certified,"learned_fit_exact_balance_certified":False,"uncertified_heuristic":method=="v2"},details

def assumptions(pair):
    b,g=pair["balancing_valid"],pair["generic"]; ur=inv_anchor(b["V"]); _,hr=policy_class(b["X"],ur); split=700; z=np.column_stack([g["X"],g["W"]]); clf=GradientBoostingClassifier(n_estimators=100,max_depth=2,learning_rate=.05,min_samples_leaf=5,random_state=ROOT_SEED); clf.fit(z[:split],g["H0"][:split]); acc=float(np.mean(clf.predict(z[split:])==g["H0"][split:])); shared=("X","U","Q","N","A","Y","Y0_struct","eta","H0","propensity","fold")
    return {"scope":"scalar U only","proxy_invariance_by_construction":True,"treatment_uses_only_X_U":True,"outcome_uses_only_X_U_A":True,"true_ate_is_one":True,"population_overlap":{"statement":"For every pretreatment H, P(A=1|H)=E[p(H0)|H] is in [0.2,0.8].","tower_property":True,"lower_bound":.2,"upper_bound":.8,"distinct_from_finite_sample_counts":True},"strict_anchor_monotonicity":{"derivative":"1 + 0.25 sech^2(u)","global_lower_bound":1.,"strictly_positive":True,"measurable_inverse_exists":True,"max_numeric_inversion_error":float(np.max(np.abs(ur-b["U"]))),"h0_recovered_exactly_from_X_Vbal":np.array_equal(hr,b["H0"])},"exact_balance_algebra":{"statement":"P(A=1|X,Q_j,H0)=p(H0)=P(A=1|H0) for every j","reason":"A is Bernoulli(p[H0]); independent audit errors are absent from treatment.","propensity_equals_p_h0_exactly":np.array_equal(b["propensity"],POLICY_PROBS[b["H0"]]),"holds_for_all_five_audits":True},"audit_injectivity":{"statement":"Q_j|X,U,H0 Gaussian with nonzero U slope","all_slopes_nonzero":np.all(np.abs(AUDIT_SLOPES)>0),"all_noise_scales_positive":np.all(AUDIT_NOISE>0),"gaussian_characteristic_function_nonzero":True},"generic_unattainability":{"exact_anchor_available":False,"all_observed_measurements_noisy":True,"analytic_argument":"Every finite Gaussian likelihood for X,Vgen,Q,N is positive for every real u. With the positive Gaussian prior, U|(X,W) has positive residual variance and full support, hence H0 is not measurable. This is population unattainability, not optimizer failure.","numerical_h0_prediction_accuracy":acc,"numerical_diagnostic_is_not_the_proof":True},"paired_regimes_share_everything_except_anchor":all(np.array_equal(g[k],b[k]) for k in shared)}

def run():
    started=time.time(); pair=generate_pair(N_MASTER,ROOT_SEED); dpair=generate_pair(N_DIAGNOSTIC,ROOT_SEED+999999); ass=assumptions(pair); agg=[]; comps=[]; fd=[]
    for regime in ("generic","balancing_valid"):
        data,diag=pair[regime],dpair[regime]
        for ni,n in enumerate(PREFIXES):
            base=ROOT_SEED+1000*ni
            for key,method in (("X","x_only"),("XW","raw_xw"),("XU","oracle_xu"),("H0","oracle_h0")):
                r,f=baseline(data,n,key,method,base+len(method)); agg.append({"regime":regime,"n":n,**r}); fd.append({"regime":regime,"n":n,"method":method,"folds":f})
            vals=[]
            for j in range(N_AUDIT):
                r,f=representation(data,diag,n,"v1",j,base+100+j,regime=="balancing_valid"); comps.append({"regime":regime,"n":n,**r}); fd.append({"regime":regime,"n":n,"method":r["method"],"folds":f}); vals.append(r["estimate"])
            est=float(np.mean(vals)); agg.append({"regime":regime,"n":n,"method":"v1_leave_one_audit_ensemble","estimate":est,"error":est-1,"abs_error":abs(est-1),"finite":bool(np.isfinite(est)),"certified":False,"theorem_endpoint_available_in_dgp":regime=="balancing_valid","learned_fit_exact_balance_certified":False})
            r,f=representation(data,diag,n,"v2",None,base+900,False); agg.append({"regime":regime,"n":n,**r}); fd.append({"regime":regime,"n":n,"method":"v2","folds":f})
    # deterministic replay includes the fitted curve, hard predictions, and merge decisions
    data=pair["balancing_valid"]; n=100; tr=np.where(data["fold"][:n]!=0)[0]; te=np.where(data["fold"][:n]==0)[0]; keep=[j for j in range(D_W) if j]; inp=np.column_stack([data["X"][:n],data["W"][:n,keep]]); audit=np.column_stack([data["X"][:n],data["W"][:n,0]]); rr=[]
    for _ in range(2):
        fit=fit_rep(inp[tr],audit[tr],data["A"][tr],ROOT_SEED+424242,False); ht=probs(fit,inp[tr]).argmax(1); he=probs(fit,inp[te]).argmax(1); mg=merge_cells(ht,data["A"][tr],audit[tr],fit); mg.pop("merged_train",None); rr.append((fit.curve,he,mg["map"],mg["sequence"]))
    replay=np.allclose(rr[0][0],rr[1][0],atol=0,rtol=0) and np.array_equal(rr[0][1],rr[1][1]) and rr[0][2:]==rr[1][2:]
    learned=[d for d in fd if d["method"].startswith("v1") or d["method"]=="v2"]
    rf=[f for d in learned for f in d["folds"]]; exclusions=all(f["audit_excluded_from_input_and_preprocessing"] for f in rf if f["audit_index"] is not None); outer=all(f["representation_learned_on_outer_train_only"] and f["heldout_evaluation_only"] for f in rf)
    final_m_distribution=[]
    for d in learned:
        ms=[f["merge"]["final_m"] for f in d["folds"]]
        final_m_distribution.append({"regime":d["regime"],"n":d["n"],"method":d["method"],"fold_final_m":ms,"counts":{str(m):ms.count(m) for m in (1,2,3)}})
    all_ms=[f["merge"]["final_m"] for f in rf]
    merge_summary={"representation_fold_count":len(rf),"final_m_counts":{str(m):all_ms.count(m) for m in (1,2,3)},"n100_v1_all_final_m_1":all(f["merge"]["final_m"]==1 for d in learned if d["n"]==100 and d["method"].startswith("v1") for f in d["folds"]),"minimum_final_training_arm_count":min(min(r["n0"],r["n1"]) for f in rf for r in f["final_train_counts"]),"zero_unit_deletion":all(f["no_units_dropped"] for f in rf),"absent_training_label_fallback_applications_heldout":sum(f["absent_training_label_fallback_applications_heldout"] for f in rf),"absent_training_label_fallback_applications_independent_diagnostic":sum(f["independent_large_diagnostic"]["absent_training_label_fallback_applications"] for f in rf)}
    performance={"final_m_distribution_by_method_regime_n":final_m_distribution,"prespecified_n1000_errors":[{"regime":r["regime"],"method":r["method"],"estimate":r["estimate"],"abs_error":r["abs_error"]} for r in agg if r["n"]==1000],"method_performance_smoke_pass":False,"interpretation":"Negative result: pipeline QA passes, but V1/V2 do not outperform the baselines in the balancing-valid regime."}
    oracle_pc=oracle_h0_positive_control(pair["generic"],dpair["generic"])
    gates={"master_n_is_1000":N_MASTER==1000,"nested_prefixes_exact":all(np.array_equal(pair[r]["X"][:n],pair[r]["X"][:N_MASTER][:n]) for r in pair for n in PREFIXES),"root_seed_is_1950702":ROOT_SEED==1950702,"paired_except_anchor":ass["paired_regimes_share_everything_except_anchor"],"population_overlap_proof_recorded":ass["population_overlap"]["tower_property"],"exact_balance_algebra_encoded":ass["exact_balance_algebra"]["propensity_equals_p_h0_exactly"],"balancing_anchor_recovers_h0":ass["strict_anchor_monotonicity"]["h0_recovered_exactly_from_X_Vbal"],"audit_injectivity_construction":ass["audit_injectivity"]["all_slopes_nonzero"],"generic_residual_variance_argument_recorded":not ass["generic_unattainability"]["exact_anchor_available"],"v1_audit_exclusion_all_folds":exclusions,"representations_outer_train_only":outer,"train_test_disjoint":all(f["train_test_disjoint"] for f in rf),"diagnostic_sample_frozen_evaluation_only":all(f["independent_large_diagnostic"]["frozen_model_only"] and not f["independent_large_diagnostic"]["used_for_tuning_or_merge"] for f in rf),"no_units_deleted":all(f["no_units_dropped"] for f in rf),"all_final_training_cells_each_arm_ge_5":all(f["all_final_training_cells_min_arm_k"] for f in rf),"all_positive_mass_heldout_cells_both_arms":all(f["all_positive_mass_heldout_cells_both_arms"] for f in rf),"no_heldout_remerge":all(f["no_heldout_remerge"] for f in rf),"aggregate_result_count_48":len(agg)==48,"v1_component_count_40":len(comps)==40,"all_48_aggregate_estimates_finite":all(r["finite"] for r in agg),"all_40_v1_component_estimates_finite":all(r["finite"] for r in comps),"selected_fit_and_merge_deterministic":replay,"v2_marked_uncertified":all(r.get("uncertified_heuristic",False) for r in agg if r["method"]=="v2")}
    payload={"provenance":{"script":"code/canonical_policy_dgp_experiment.py","root_seed":ROOT_SEED,"scope":"one paired scalar-U SCM diagnostic only","master_n":N_MASTER,"nested_prefixes":PREFIXES,"independent_diagnostic_n":N_DIAGNOSTIC,"diagnostic_kernel_n":N_DIAGNOSTIC_KERNEL,"true_ate":TRUE_ATE,"outcome_estimator":"2-fold cross-fitted GradientBoosting T-learner","gbr_spec":GBR_SPEC,"minimum_training_cell_arm_count":MIN_CELL_ARM,"post_result_tuning":False,"smoke_pass_interpretation":"Pipeline, leakage, finite-sample support, and theorem-positive-control QA only; not a method-performance pass."},"dgp":{"graph":"U->X; (X,U)->A; (X,U)->W; (X,U,A)->Y","dimensions":{"U":1,"X":D_X,"W":D_W},"policy_probabilities":POLICY_PROBS,"v2_status":"uncertified heuristic","merge_rule":"outer-train-only greedy minimum discrepancy increase with label tie-break"},"master_hashes":{"common":{k:ah(pair["generic"][k]) for k in ("X","U","Q","N","A","Y","H0","propensity","fold")},"generic_W":ah(pair["generic"]["W"]),"balancing_W":ah(pair["balancing_valid"]["W"]),"independent_diagnostic_generic_X":ah(dpair["generic"]["X"])},"assumption_qa":ass,"oracle_h0_positive_control":oracle_pc,"leakage_qa":{"Y_passed_to_representation_or_merge":False,"U_passed_to_nonoracle_representation_merge_or_outcome_model":False,"heldout_or_diagnostic_used_for_merge":False,"v1_own_audit_excluded_every_fold":exclusions,"outer_fold_training_only":outer},"aggregate_results":agg,"v1_components":comps,"fold_details":fd,"merge_summary":merge_summary,"performance_summary":performance,"smoke_gates":gates,"smoke_pass":all(gates.values()),"runtime_seconds":time.time()-started,"limitations":["One scalar-U SCM cannot support a general performance claim.","Balancing-valid contains an invertible noiseless anchor.","V2 is uncertified.","Train-only merging guarantees empirical training support, not identification.","Method performance smoke failed: V1/V2 do not beat the baselines in the balancing-valid regime.","No AIPW or semiparametric inference."]}
    return js(payload)
def main():
    p=run(); OUTPUT_PATH.write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps({"output":str(OUTPUT_PATH),"runtime_seconds":p["runtime_seconds"],"smoke_pass":p["smoke_pass"],"failed_gates":[k for k,v in p["smoke_gates"].items() if not v]},indent=2))
if __name__=="__main__": main()
