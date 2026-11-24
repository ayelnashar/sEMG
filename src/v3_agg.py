"""Aggregate all revised experiments into N.pkl (numbers for text, tables, figures)."""
import sys, glob, pickle, json, numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
from scipy.stats import ttest_1samp, ttest_rel
from core import aurc
E = RES
ld = lambda pat: [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{E}/{pat}"))]
A = ld("v3_a/a_*.pkl"); S = ld("v3_s/s_*.pkl"); import os; NV = ld(os.environ.get("NVD", "v3_n") + "/n_*.pkl"); W = ld(os.environ.get("WD", "v3_w") + "/w_*.pkl")
B = {tc: ld(f"v3_b/{tc}_*.pkl") for tc in ["rectified", "signed", "mixed"]}
LDA = pickle.load(open(f"{E}/v3_lda.pkl", "rb"))
N = dict(nA=len(A), nS=len(S), nNV=len(NV), nW=len(W), nB={k: len(v) for k, v in B.items()})
ms = lambda v: (float(np.nanmean(v)), float(np.nanstd(v)))
mm = lambda v: (float(np.nanmin(v)), float(np.nanmax(v)))
STATS = ["density", "confidence", "entropy", "margin", "energy"]

# ---- ladder (three-class) and six-class
N["mismatch_seed_raw"] = {k: [r["ladder3"]["mismatch"]["auroc"][k] for r in A] for k in ["confidence", "density"]}
N["ladder3"] = {c: {k: ms([r["ladder3"][c][k] for r in A]) for k in ["acc", "bal", "mauc", "ece", "conf", "refusal"]}
                for c in A[0]["ladder3"]}
for c in A[0]["ladder3"]:
    if "auroc" in A[0]["ladder3"][c]:
        N["ladder3"][c]["auroc"] = {k: ms([r["ladder3"][c]["auroc"][k] for r in A]) for k in A[0]["ladder3"][c]["auroc"]}
        N["ladder3"][c]["auroc_range"] = {k: mm([r["ladder3"][c]["auroc"][k] for r in A]) for k in A[0]["ladder3"][c]["auroc"]}
        N["ladder3"][c]["tpr5"] = {k: ms([r["ladder3"][c]["tpr5"][k] for r in A]) for k in STATS}
    for k in ["bal_accepted", "bal_refused", "err_accepted", "err_refused"]:
        if k in A[0]["ladder3"][c]: N["ladder3"][c][k] = ms([r["ladder3"][c].get(k, np.nan) for r in A])
    N["ladder3"][c]["recall"] = np.nanmean([r["ladder3"][c]["recall"] for r in A], 0).tolist()
    N["ladder3"][c]["pred_hist"] = np.mean([r["ladder3"][c]["pred_hist"] for r in A], 0).tolist()
N["six"] = {c: {k: ms([r[c][k] for r in A]) for k in ["acc", "bal", "mauc", "ece", "conf", "refusal",
                                                         "bal_accepted", "bal_refused", "err_accepted", "err_refused"]}
            for c in ["source_test", "target"]}
N["recall6"] = {c: ms([r[c]["recall"] for r in A]) for c in ["source_test", "target"]}
N["recall6"] = {c: (np.mean([r[c]["recall"] for r in A], 0).tolist(), np.std([r[c]["recall"] for r in A], 0).tolist())
                for c in ["source_test", "target"]}
N["T"] = mm([r["T"] for r in A])
N["aurc"] = {k: ms([r["target"]["aurc"][k] for r in A]) for k in STATS}
N["untrained"] = dict(acc=ms([r["untrained"]["acc"] for r in A]),
                      gap=ms([r["untrained"]["aurc"]["density"] - r["untrained"]["aurc"]["confidence"] for r in A]))
N["pre_gap"] = ms([r["target"]["aurc"]["density"] - r["target"]["aurc"]["confidence"] for r in A])
N["latency"] = A[0].get("latency")
pp = {j: [r["per_participant"][j]["acc"] for r in A] for j in range(21, 31)}
N["participant_acc"] = mm([np.mean(v) for v in pp.values()]) + (float(np.std([np.mean(v) for v in pp.values()])),)

# ---- shifts on Dataset A (six-class)
N["shift"] = {}
for c in A[0]["shift"]:
    d = {k: ms([r["shift"][c][k] for r in A]) for k in ["bal", "conf", "refusal"]}
    d["auroc"] = {k: ms([r["shift"][c]["auroc"][k] for r in A]) for k in A[0]["shift"][c]["auroc"]}
    d["tpr5"] = {k: ms([r["shift"][c]["tpr5"][k] for r in A]) for k in STATS}
    if c in LDA["A"]["shift"]:
        d["lda_bal"] = (LDA["A"]["shift"][c]["bal"], 0.0)
        d["lda_auroc"] = {k: (LDA["A"]["shift"][c]["auroc"][k], 0.0) for k in ["confidence", "density"]}
    N["shift"][c] = d
N["lda_source_test_bal6"] = (LDA["A"]["source_test_bal6"], 0.0)
N["lda_ladder3"] = {c: dict(bal=(v["bal"], 0.0), auroc={k: (v["auroc"][k], 0.0) for k in ["confidence", "density"]} if "auroc" in v else None)
                    for c, v in LDA["A"]["ladder3"].items()}

# ---- novelty
N["novelty"] = {k: ms([r["auroc"][k] for r in NV]) for k in STATS}
N["novelty_pairs"] = {str(p): {k: ms([r["auroc"][k] for r in NV if tuple(r["pair"]) == p]) for k in STATS}
                      for p in sorted({tuple(r["pair"]) for r in NV})}

# ---- within B
N["B"] = {}
for tc, rs in B.items():
    if not rs: continue
    N["B"][tc] = {}
    for c in rs[0]["cond"]:
        d = {k: ms([r["cond"][c][k] for r in rs]) for k in ["bal", "acc", "conf", "refusal"]}
        if "auroc" in rs[0]["cond"][c]:
            d["auroc"] = {k: ms([r["cond"][c]["auroc"][k] for r in rs]) for k in rs[0]["cond"][c]["auroc"]}
            d["auroc_range"] = {k: mm([r["cond"][c]["auroc"][k] for r in rs]) for k in rs[0]["cond"][c]["auroc"]}
            d["auroc_raw"] = {k: [r["cond"][c]["auroc"][k] for r in rs] for k in ["confidence", "density"]}
            d["n_below"] = {k: int(sum(r["cond"][c]["auroc"][k] < 0.5 for r in rs)) for k in rs[0]["cond"][c]["auroc"]}
        ob = LDA["B"][tc]
        if ob and c in ob[0]:
            d["lda_bal"] = ms([o[c]["bal"] for o in ob])
            if "auroc" in ob[0][c]:
                d["lda_auroc"] = {k: ms([o[c]["auroc"][k] for o in ob]) for k in ["confidence", "density"]}
        N["B"][tc][c] = d
    N["B"][tc]["_counts"] = np.mean([r["counts_test"] for r in rs], 0).tolist()

# ---- splits
if S:
    N["splits"] = {}
    for c in ["cross-dataset", "mismatch"]:
        N["splits"][c] = dict(bal=ms([r[c]["bal"] for r in S]), conf=ms([r[c]["conf"] for r in S]),
                              auroc={k: ms([r[c]["auroc"][k] for r in S]) for k in STATS},
                              auroc_range={k: mm([r[c]["auroc"][k] for r in S]) for k in STATS},
                              auroc_raw={k: [r[c]["auroc"][k] for r in S] for k in STATS})
    N["splits"]["acc"] = ms([r["acc"] for r in S]); N["splits"]["bal"] = ms([r["bal"] for r in S])
    taps = ["proj", "layer1", "layer2", "layer3_norm"]
    N["depth"] = {t: dict(bal=ms([r["depth"][t]["bal"] for r in S]), acc=ms([r["depth"][t]["acc"] for r in S]),
                          ood=ms([r["depth"][t]["mean_ood"] for r in S]), aurc=ms([r["depth"][t]["aurc_conf"] for r in S]),
                          fault={f: ms([r["depth"][t]["ood"][f] for r in S]) for f in S[0]["depth"][t]["ood"]}) for t in taps}
    N["depth_win"] = int(sum(r["depth"]["layer2"]["mean_ood"] > r["depth"]["layer3_norm"]["mean_ood"] for r in S))

# ---- 200 ms
if W:
    N["w200"] = {c: dict(bal=ms([r["cond"][c]["bal"] for r in W]), conf=ms([r["cond"][c]["conf"] for r in W]),
                         acc=ms([r["cond"][c]["acc"] for r in W]),
                         auroc={k: ms([r["cond"][c]["auroc"][k] for r in W]) for k in STATS} if "auroc" in W[0]["cond"][c] else None,
                         auroc_range={k: mm([r["cond"][c]["auroc"][k] for r in W]) for k in STATS} if "auroc" in W[0]["cond"][c] else None)
                 for c in W[0]["cond"]}
    N["w200_n"] = W[0]["n_windows"]

# ---- cascade
N["cascade"] = {k: {m: ms([r["cascade"][k][m] for r in A]) for m in A[0]["cascade"][k]} for k in A[0]["cascade"]}
N["cascade_meta"] = {k: ms([r["cascade_meta"][k] for r in A]) for k in A[0]["cascade_meta"]}

# ---- statistics
tests = {}
if S:
    v = [r["mismatch"]["auroc"]["confidence"] for r in S]; tests["Confidence inverted under mismatch, A to B"] = ttest_1samp(v, 0.5).pvalue
    v = [r["mismatch"]["auroc"]["density"] for r in S]; tests["Density detects mismatch, A to B"] = ttest_1samp(v, 0.5).pvalue
    v = [r["cross-dataset"]["bal"] for r in S]; tests["Cross-dataset balanced acc. above chance"] = ttest_1samp(v, 1 / 3).pvalue
    a = [r["depth"]["layer2"]["mean_ood"] for r in S]; b = [r["depth"]["layer3_norm"]["mean_ood"] for r in S]
    tests["Detection, block 2 vs output"] = ttest_rel(a, b).pvalue
    a = [r["depth"]["layer3_norm"]["bal"] for r in S]; b = [r["depth"]["proj"]["bal"] for r in S]
    tests["Balanced acc., output vs projection"] = ttest_rel(a, b).pvalue
if B["rectified"]:
    v = [r["cond"]["signed (no rectification)"]["auroc"]["confidence"] for r in B["rectified"]]
    tests["Confidence inverted under mismatch, within B"] = ttest_1samp(v, 0.5).pvalue
    v = [r["cond"]["signed (no rectification)"]["auroc"]["density"] for r in B["rectified"]]
    tests["Density detects mismatch, within B"] = ttest_1samp(v, 0.5).pvalue
a = [r["cascade"]["range + density + confidence"]["harmful"] for r in A]; b = [r["cascade"]["confidence"]["harmful"] for r in A]
tests["Harmful acceptance, layered vs confidence"] = ttest_rel(a, b).pvalue
# participant-clustered bootstrap for error ranking, averaged over seeds
rng = np.random.RandomState(0); parts = A[0]["target"]["part"] if "part" in A[0]["target"] else None
import core
TG = core.IDX["target"]; part = core.s1[TG]; uniq = np.unique(part); idxs = {u: np.where(part == u)[0] for u in uniq}
diffs = {k: [] for k in ["density", "entropy", "margin", "energy"]}
for _ in range(2000):
    pick = rng.choice(uniq, len(uniq), replace=True); ix = np.concatenate([idxs[u] for u in pick])
    acc = {k: 0.0 for k in diffs}
    for r in A:
        sc = r["target"]["scores"]; cr = r["target"]["correct"]
        ref = aurc(sc["confidence"][ix], cr[ix])
        for k in diffs: acc[k] += aurc(sc[k][ix], cr[ix]) - ref
    for k in diffs: diffs[k].append(acc[k] / len(A))
N["boot"] = {}
for k, dd in diffs.items():
    dd = np.array(dd); lo, md, hi = np.percentile(dd, [2.5, 50, 97.5])
    p = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / 2001)
    N["boot"][k] = dict(d=float(md), lo=float(lo), hi=float(hi), p=float(p))
    tests[f"{k.capitalize()} vs max softmax, error ranking"] = p
names = list(tests); p = np.array([tests[k] for k in names]); m = len(p)
o = np.argsort(p); q = np.empty(m); prev = 1.0
for rank, i in reversed(list(enumerate(o, 1))):
    prev = min(prev, p[i] * m / rank); q[i] = prev
N["tests"] = [dict(name=names[i], p=float(p[i]), bonf=float(min(1, p[i] * m)), q=float(q[i])) for i in o]
pickle.dump(N, open(f"{E}/N.pkl", "wb"))
print(json.dumps({k: N[k] for k in ["nA", "nS", "nNV", "nW", "nB"]}))
for t in N["tests"]: print(f"  {t['name']:48s} p={t['p']:.2e} q={t['q']:.2e} bonf={t['bonf']:.2e}")
# per-participant cross-participant results (Table XIII)
pp = {j: {k: ms([r["per_participant"][j][k] for r in A]) for k in ["acc", "bal", "conf"]} for j in range(21, 31)}
pickle.dump(pp, open(f"{E}/v3_pp.pkl", "wb"))
