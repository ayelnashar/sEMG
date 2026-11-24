"""Classical baseline: Hudgins time-domain features with shrinkage LDA, on Dataset A
(source test, mismatch family, three-class ladder) and within Dataset B (ten splits).
Deterministic; no neural network involved."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
import pickle
from core import *
from lib2 import make_conditions, td_features, LDABaseline
from lib3 import auroc
from sklearn.metrics import balanced_accuracy_score as BA

is_cal = np.isin(s1[ST], CALIB_SUBJECTS); FIT = ST[~is_cal]; TE = IDX["source_test"]
CONDS = make_conditions(lambda X: X, MU, SD, False, True); CONDS.pop("matched")
R = {"A": {}, "B": {}}
lda = LDABaseline().fit(td_features(X1[FIT]), y1[FIT], K6)
p0, s0 = lda.run(td_features(X1[TE]))
R["A"]["source_test_bal6"] = float(BA(y1[TE], p0.argmax(1)))
R["A"]["shift"] = {}
for nm, f in CONDS.items():
    p, s = lda.run(td_features(f(X1[TE]) * SD + MU))
    R["A"]["shift"][nm] = dict(bal=float(BA(y1[TE], p.argmax(1))), auroc={k: auroc(s0[k], s[k]) for k in s})
st3 = sub3("source_train"); ic = np.isin(s1[st3], CALIB_SUBJECTS)
lda3 = LDABaseline().fit(td_features(X1[st3[~ic]]), ym[st3[~ic]], KC)
L = {}; ref = None
for nm, Xraw, yq in [("source_test", X1[sub3("source_test")], ym[sub3("source_test")]),
                     ("cross-participant", X1[sub3("target")], ym[sub3("target")]),
                     ("cross-dataset", np.abs(X2), y2), ("mismatch", X2, y2)]:
    p, s = lda3.run(td_features(Xraw)); r = dict(bal=float(BA(yq, p.argmax(1))))
    if ref is None: ref = s
    else: r["auroc"] = {k: auroc(ref[k], s[k]) for k in s}
    L[nm] = r
R["A"]["ladder3"] = L
pids = np.unique(s2)
for tc in ["rectified", "signed", "mixed"]:
    R["B"][tc] = []
    for seed in [int(x) for x in os.environ.get("SEEDS_B", "0,1,2,3,4,5,6,7,8,9").split(",")]:
        perm = np.random.RandomState(100 + seed).permutation(pids)
        tr_p, te_p = perm[:26], perm[26:]; cal_p = tr_p[-4:]
        itr = np.where(np.isin(s2, tr_p))[0]; ite = np.where(np.isin(s2, te_p))[0]
        fit = itr[~np.isin(s2[itr], cal_p)]
        if tc == "mixed":
            conv = np.abs; Xtr_c = np.concatenate([np.abs(X2[itr]), X2[itr]])
            F = np.concatenate([td_features(np.abs(X2[fit])), td_features(X2[fit])]); yf = np.r_[y2[fit], y2[fit]]
        else:
            conv = np.abs if tc == "rectified" else (lambda X: X); Xtr_c = conv(X2[itr])
            F = td_features(conv(X2[fit])); yf = y2[fit]
        mu, sd = norm_fit(Xtr_c)
        conds = make_conditions(conv, mu, sd, True, tc != "signed")
        if tc == "mixed": conds["signed (no rectification)"] = lambda X: ((X - mu) / sd).astype(np.float32)
        lb = LDABaseline().fit(F, yf, KC); yt = y2[ite]; out = {}; refs = None
        for nm, f in conds.items():
            p, s = lb.run(td_features(f(X2[ite]) * sd + mu)); r = dict(bal=float(BA(yt, p.argmax(1))))
            if nm == "matched": refs = s
            else: r["auroc"] = {k: auroc(refs[k], s[k]) for k in s}
            out[nm] = r
        R["B"][tc].append(out)
pickle.dump(R, open(os.path.join(RES, "v3_lda.pkl"), "wb"))
print("LDA source test (6 classes):", round(R["A"]["source_test_bal6"], 3),
      "| within B signed:", round(np.mean([o["signed (no rectification)"]["bal"] for o in R["B"]["rectified"]]), 3))
