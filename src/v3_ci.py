"""Participant-clustered bootstrap intervals, from the saved main-split encoders."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
import pickle
from core import *
from lib3 import fit_probe, pstats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
OUT = os.path.join(RES, "v3_a")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46,47,48,49,50,51").split(",")]
NB = int(os.environ.get("NBOOT", 2000))
st3, tg3 = sub3("source_test"), sub3("target")
per_seed = []
for seed in SEEDS:
    enc = EMGEncoder().to(DEV)
    enc.load_state_dict(torch.load(f"{OUT}/enc_{seed}.pt", map_location=DEV))
    Zr = embed(enc, norm(X1)); m0, s0 = Zr[ST].mean(0), Zr[ST].std(0) + 1e-6
    Zs = lambda Xin: ((embed(enc, Xin) - m0) / s0).astype(np.float32); Z = ((Zr - m0) / s0).astype(np.float32)
    tr3 = sub3("source_train"); ic = np.isin(s1[tr3], CALIB_SUBJECTS)
    b3 = fit_probe(Z[tr3[~ic]], ym[tr3[~ic]], Z[tr3[ic]], ym[tr3[ic]], KC, seed)
    g3 = Gate().fit(Z[tr3[~ic]], ym[tr3[~ic]], KC)
    r = {}
    for nm, Zq, yq in [("source_test", Z[st3], ym[st3]), ("cross-participant", Z[tg3], ym[tg3]),
                       ("cross-dataset", Zs(norm(np.abs(X2))), y2), ("mismatch", Zs(norm(X2)), y2)]:
        p, sc = pstats(b3, g3, Zq); r[nm] = dict(pred=p.argmax(1), y=yq, conf=sc["confidence"], dens=sc["density"])
    per_seed.append(r)
    print("seed", seed, "done", flush=True)
gA_test, gA_tgt, gB = s1[st3], s1[tg3], s2
uT, uG, uB = np.unique(gA_test), np.unique(gA_tgt), np.unique(gB)
ix = lambda g, u, pick: np.concatenate([np.where(g == k)[0] for k in pick])
rng = np.random.RandomState(0)
boot = {k: [] for k in ["bal_cp", "bal_cd", "bal_mm", "auc_conf", "auc_dens"]}
for _ in range(NB):
    iT = ix(gA_test, uT, rng.choice(uT, len(uT))); iG = ix(gA_tgt, uG, rng.choice(uG, len(uG)))
    iB = ix(gB, uB, rng.choice(uB, len(uB)))
    v = {k: [] for k in boot}
    for r in per_seed:
        v["bal_cp"].append(balanced_accuracy_score(r["cross-participant"]["y"][iG], r["cross-participant"]["pred"][iG]))
        v["bal_cd"].append(balanced_accuracy_score(r["cross-dataset"]["y"][iB], r["cross-dataset"]["pred"][iB]))
        v["bal_mm"].append(balanced_accuracy_score(r["mismatch"]["y"][iB], r["mismatch"]["pred"][iB]))
        lab = np.r_[np.zeros(len(iT)), np.ones(len(iB))]
        v["auc_conf"].append(roc_auc_score(lab, np.r_[r["source_test"]["conf"][iT], r["mismatch"]["conf"][iB]]))
        v["auc_dens"].append(roc_auc_score(lab, np.r_[r["source_test"]["dens"][iT], r["mismatch"]["dens"][iB]]))
    for k in boot: boot[k].append(np.mean(v[k]))
CI = {k: tuple(np.percentile(v, [2.5, 97.5]).tolist()) for k, v in boot.items()}
pickle.dump(CI, open(os.path.join(RES, "v3_ci.pkl"), "wb"))
print(CI)
