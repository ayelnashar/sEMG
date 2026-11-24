"""Within-Dataset-B conditioning, ten participant splits, calibrated probe."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
import time, pickle
from core import *
from lib2 import make_conditions, contract_score
from lib3 import *
OUT = os.environ.get("OUT", os.path.join(RES, "v3_b")); os.makedirs(OUT, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
pids = np.unique(s2)
def run(seed, tc):
    fn = f"{OUT}/{tc}_{seed}.pkl"
    if os.path.exists(fn): return
    t0 = time.time()
    perm = np.random.RandomState(100 + seed).permutation(pids)
    tr_p, te_p = perm[:26], perm[26:]; cal_p = tr_p[-4:]
    itr = np.where(np.isin(s2, tr_p))[0]; ite = np.where(np.isin(s2, te_p))[0]
    cal = np.isin(s2[itr], cal_p); fit, cl = itr[~cal], itr[cal]
    if tc == "mixed":
        Xtr_c = np.concatenate([np.abs(X2[itr]), X2[itr]]); conv = np.abs; ep = max(1, int(np.ceil(EPOCHS / 2)))
    else:
        conv = np.abs if tc == "rectified" else (lambda X: X); Xtr_c = conv(X2[itr]); ep = EPOCHS
    mu, sd = norm_fit(Xtr_c); lo, hi = float(Xtr_c.min()), float(Xtr_c.max())
    N = lambda X: ((X - mu) / sd).astype(np.float32)
    enc, rec = pretrain(N(Xtr_c), epochs=ep, seed=seed)
    Ztr = embed(enc, N(Xtr_c)); m0, s0 = Ztr.mean(0), Ztr.std(0) + 1e-6
    Zs = lambda Xin: ((embed(enc, Xin) - m0) / s0).astype(np.float32)
    if tc == "mixed":
        Zf = np.concatenate([Zs(N(np.abs(X2[fit]))), Zs(N(X2[fit]))]); yf = np.r_[y2[fit], y2[fit]]
        Zc = np.concatenate([Zs(N(np.abs(X2[cl]))), Zs(N(X2[cl]))]); yc = np.r_[y2[cl], y2[cl]]
    else:
        Zf, yf, Zc, yc = Zs(N(conv(X2[fit]))), y2[fit], Zs(N(conv(X2[cl]))), y2[cl]
    base = fit_probe(Zf, yf, Zc, yc, KC, seed); gate = Gate().fit(Zf, yf, KC)
    conds = make_conditions(conv, mu, sd, True, tc != "signed")
    if tc == "mixed": conds["signed (no rectification)"] = lambda X: N(X)
    yt = y2[ite]; R = dict(seed=seed, train=tc, test=te_p.tolist(), n_test=len(ite),
                           n_fit=len(fit), n_cal=len(cl), cond={}, counts_test=np.bincount(yt, minlength=3).tolist(),
                           epochs=ep, optimizer_updates=ep*int(np.ceil(len(Xtr_c)/BATCH)),
                           reference_updates=EPOCHS*int(np.ceil(len(itr)/BATCH)),
                           train_participants=tr_p.tolist(), calibration_participants=cal_p.tolist())
    ref = None
    for nm, f in conds.items():
        Xin = f(X2[ite]); Z = Zs(Xin); p, sc = pstats(base, gate, Z)
        sc["range check"] = contract_score(Xin, mu, sd, lo, hi)
        r = summary(p, yt, KC, gate, Z)
        if nm == "matched": ref = sc
        else:
            r["auroc"] = {k: auroc(ref[k], sc[k]) for k in sc}
            r["tpr5"] = {k: tpr_at(ref[k], sc[k]) for k in STATS}
        R["cond"][nm] = r
    pickle.dump(R, open(fn, "wb"))
    key = {"rectified": "signed (no rectification)", "signed": "rectified", "mixed": "gain x0.1"}[tc]
    c = R["cond"][key]
    print(f"[B {tc} {seed}] {time.time()-t0:.0f}s matched {R['cond']['matched']['bal']:.3f} | {key} bal {c['bal']:.3f} "
          f"conf {c['auroc']['confidence']:.3f} dens {c['auroc']['density']:.3f}", flush=True)
for s in SEEDS:
    for tc in ["rectified", "signed", "mixed"]: run(s, tc)
print("B DONE", flush=True)
