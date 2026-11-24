"""200 ms decision windows, five seeds, calibrated probe."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
import time, pickle
from core import *
from lib3 import *
OUT = os.environ.get("OUT", os.path.join(RES, "v3_w")); os.makedirs(OUT, exist_ok=True)
WIN = N_TOK_S * PATCH_S
def rewindow(X, *vals):
    n = X.shape[1] // WIN
    return [X[:, :n * WIN].reshape(len(X) * n, WIN, N_CH)] + [np.repeat(v, n) for v in vals]
Xs1, ys1, sj1, se1 = rewindow(X1, y1, s1, ss1); Xs2, ys2 = rewindow(X2, y2)
keepS = np.isin(ys1, list(D1_MAP)); ymS = np.array([D1_MAP.get(v, -1) for v in ys1])
itr = np.where(np.isin(sj1, range(1, 21)) & np.isin(se1, [1, 2, 3, 4]))[0]
ite = np.where(np.isin(sj1, range(1, 21)) & (se1 == 5))[0]; itg = np.where(sj1 > 20)[0]
mu_, sd_ = norm_fit(Xs1[itr]); nz = lambda X: ((X - mu_) / sd_).astype(np.float32); XN = nz(Xs1)
for seed in [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]:
    fn = f"{OUT}/w_{seed}.pkl"
    if os.path.exists(fn): continue
    t0 = time.time()
    enc, _ = pretrain(XN[itr], seed=seed, n_tok=N_TOK_S, patch=PATCH_S)
    Zr = embed(enc, XN); a0, b0 = Zr[itr].mean(0), Zr[itr].std(0) + 1e-6
    Zn = lambda A_: ((A_ - a0) / b0).astype(np.float32); ZS = Zn(Zr)
    tr3 = itr[keepS[itr]]; cal = np.isin(sj1[tr3], CALIB_SUBJECTS)
    b = fit_probe(ZS[tr3[~cal]], ymS[tr3[~cal]], ZS[tr3[cal]], ymS[tr3[cal]], KC, seed)
    g = Gate().fit(ZS[tr3[~cal]], ymS[tr3[~cal]], KC)
    R = dict(seed=seed, n_windows=dict(A=len(Xs1), B=len(Xs2)), cond={}); ref = None
    for nm, Zq, yq in [("source_test", ZS[ite[keepS[ite]]], ymS[ite[keepS[ite]]]),
                       ("cross-participant", ZS[itg[keepS[itg]]], ymS[itg[keepS[itg]]]),
                       ("cross-dataset", Zn(embed(enc, nz(np.abs(Xs2)))), ys2),
                       ("mismatch", Zn(embed(enc, nz(Xs2))), ys2)]:
        p, sc = pstats(b, g, Zq); r = summary(p, yq, KC, g, Zq); r["n"] = len(yq)
        if ref is None: ref = sc
        else: r["auroc"] = {k: auroc(ref[k], sc[k]) for k in STATS}
        R["cond"][nm] = r
    pickle.dump(R, open(fn, "wb"))
    m = R["cond"]["mismatch"]
    print(f"[200ms {seed}] {time.time()-t0:.0f}s mismatch bal {m['bal']:.3f} conf {m['conf']:.4f} AUROC conf {m['auroc']['confidence']:.3f} dens {m['auroc']['density']:.3f}", flush=True)
print("W DONE", flush=True)
