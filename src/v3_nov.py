"""Held-out gestures excluded from encoder pretraining: four pairs x five seeds, probe."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
import time, pickle
from core import *
from lib3 import *
OUT = os.environ.get("OUT", os.path.join(RES, "v3_n")); os.makedirs(OUT, exist_ok=True)
PAIRS = [(4, 5), (1, 2), (3, 4), (2, 5)][:int(os.environ.get("NPAIRS", 4))]
for seed in [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]:
    for hp in PAIRS:
        fn = f"{OUT}/n_{seed}_{hp[0]}{hp[1]}.pkl"
        if os.path.exists(fn): continue
        t0 = time.time(); seen = [c for c in range(K6) if c not in hp]
        trs = ST[np.isin(y1[ST], seen)]
        mu_, sd_ = norm_fit(X1[trs]); Xr = ((X1 - mu_) / sd_).astype(np.float32)
        e_, _ = pretrain(Xr[trs], seed=seed)
        Zr = embed(e_, Xr); Z = ((Zr - Zr[trs].mean(0)) / (Zr[trs].std(0) + 1e-6)).astype(np.float32)
        rm = {c: i for i, c in enumerate(seen)}; cal = np.isin(s1[trs], CALIB_SUBJECTS); fi, ci = trs[~cal], trs[cal]
        b_ = fit_probe(Z[fi], np.array([rm[c] for c in y1[fi]]), Z[ci], np.array([rm[c] for c in y1[ci]]), 4, seed)
        g_ = Gate().fit(Z[fi], np.array([rm[c] for c in y1[fi]]), 4)
        te = IDX["source_test"]; inm = np.isin(y1[te], seen)
        _, si = pstats(b_, g_, Z[te[inm]]); _, so = pstats(b_, g_, Z[te[~inm]])
        R = dict(seed=seed, pair=hp, n_in=int(inm.sum()), n_out=int((~inm).sum()),
                 auroc={k: auroc(si[k], so[k]) for k in STATS})
        pickle.dump(R, open(fn, "wb"))
        print(f"[nov {seed} {hp}] {time.time()-t0:.0f}s dens {R['auroc']['density']:.3f} conf {R['auroc']['confidence']:.3f}", flush=True)
print("N DONE", flush=True)
