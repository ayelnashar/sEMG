"""Ten random 20/10 participant splits: depth and conditioning mismatch, calibrated probe."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
import time, pickle
from core import *
import core
from lib2 import contract_score
from lib3 import *
OUT = os.environ.get("OUT", os.path.join(RES, "v3_s")); os.makedirs(OUT, exist_ok=True)
FAULT_FN = dict(core.FAULTS)
for r in range(int(os.environ.get("NS", 10))):
    fn = f"{OUT}/s_{r}.pkl"
    if os.path.exists(fn): continue
    t0 = time.time()
    perm = np.random.RandomState(1000 + r).permutation(np.arange(1, 31))
    tr_p, te_p = perm[:20], perm[20:]; cal_p = tr_p[-4:]
    itr = np.where(np.isin(s1, tr_p) & np.isin(ss1, [1, 2, 3, 4]))[0]
    ite = np.where(np.isin(s1, te_p))[0]; iref = np.where(np.isin(s1, tr_p) & (ss1 == 5))[0]
    mu_, sd_ = norm_fit(X1[itr]); Nf = lambda X: ((X - mu_) / sd_).astype(np.float32); Xr = Nf(X1)
    enc, _ = pretrain(Xr[itr], seed=42 + r)
    isc = np.isin(s1[itr], cal_p); fi, ci = itr[~isc], itr[isc]
    R = dict(split=r, test=te_p.tolist(), cal=cal_p.tolist(), depth={})
    core.RNG = np.random.RandomState(r)
    faulted = {nm: Nf(f(X1[iref])) for nm, f in FAULT_FN.items()}
    for tap in ["proj", "layer1", "layer2", "layer3_norm"]:
        Zt = embed(enc, Xr, tap=tap); mt, st_ = Zt[itr].mean(0), Zt[itr].std(0) + 1e-6
        Zt = ((Zt - mt) / st_).astype(np.float32)
        bt = fit_probe(Zt[fi], y1[fi], Zt[ci], y1[ci], K6, 42 + r); gt = Gate().fit(Zt[fi], y1[fi], K6)
        pin, sin = pstats(bt, gt, Zt[iref]); oods = {}
        for nm, Xf in faulted.items():
            Zo = ((embed(enc, Xf, tap=tap) - mt) / st_).astype(np.float32)
            oods[nm] = auroc(sin["density"], gt.score(Zo))
        pt, stt = pstats(bt, gt, Zt[ite]); pr = pt.argmax(1); corr = (pr == y1[ite]).astype(float)
        R["depth"][tap] = dict(acc=float(corr.mean()), bal=float(balanced_accuracy_score(y1[ite], pr)),
                               ood=oods, mean_ood=float(np.mean(list(oods.values()))),
                               aurc_conf=aurc(stt["confidence"], corr))
    d = R["depth"]["layer3_norm"]; R["acc"] = d["acc"]; R["bal"] = d["bal"]
    Zr = embed(enc, Xr); m0, s0 = Zr[itr].mean(0), Zr[itr].std(0) + 1e-6
    Zs = lambda Xin: ((embed(enc, Xin) - m0) / s0).astype(np.float32); Z = ((Zr - m0) / s0).astype(np.float32)
    k3 = keep[itr]; f3, c3 = itr[k3 & ~isc], itr[k3 & isc]
    b3 = fit_probe(Z[f3], ym[f3], Z[c3], ym[c3], KC, 42 + r); g3 = Gate().fit(Z[f3], ym[f3], KC)
    r3 = iref[keep[iref]]; _, sref = pstats(b3, g3, Z[r3])
    for nm, Xq in [("cross-dataset", Nf(np.abs(X2))), ("mismatch", Nf(X2))]:
        Zq = Zs(Xq); p3, s3 = pstats(b3, g3, Zq)
        R[nm] = dict(bal=float(balanced_accuracy_score(y2, p3.argmax(1))), conf=float(p3.max(1).mean()),
                     auroc={k: auroc(sref[k], s3[k]) for k in s3})
    pickle.dump(R, open(fn, "wb"))
    print(f"[split {r}] {time.time()-t0:.0f}s ood " + " ".join(f"{k}:{v['mean_ood']:.3f}" for k, v in R["depth"].items())
          + f" | mismatch bal {R['mismatch']['bal']:.3f} conf {R['mismatch']['auroc']['confidence']:.3f}", flush=True)
print("S DONE", flush=True)
