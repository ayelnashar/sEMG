"""Write Tables III to XII of the manuscript as CSV files and print the headline numbers."""
import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
N = pickle.load(open(os.path.join(RES, "N.pkl"), "rb"))
OUT = os.path.join(RES, "tables"); os.makedirs(OUT, exist_ok=True)
f3 = lambda t: f"{t[0]:.3f} ± {t[1]:.3f}"; g3 = lambda t: f"{t[0]:.3f}"
L, S, B, C, D = N["ladder3"], N["shift"], N["B"], N["cascade"], N["depth"]
RB, SB, XB = B["rectified"], B["signed"], B["mixed"]
def save(name, cols, rows):
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(OUT, name + ".csv"), index=False)

gap = lambda d: f"{100 * (d['bal_accepted'][0] - d['bal_refused'][0]):.1f}"
save("table03_ladder", ["Partition", "Pooled acc.", "Balanced acc.", "Refusal", "Mean conf.", "Refused gap (points)"],
     [[n, g3(L[c]["acc"]), f3(L[c]["bal"]), g3(L[c]["refusal"]), g3(L[c]["conf"]), gap(L[c])] for n, c in
      [("Source test", "source_test"), ("Cross-participant", "cross-participant"), ("Cross-dataset", "cross-dataset"),
       ("Conditioning mismatch", "mismatch")]] +
     [[f"Six-class {n}", g3(N["six"][c]["acc"]), f3(N["six"][c]["bal"]), g3(N["six"][c]["refusal"]), g3(N["six"][c]["conf"]), gap(N["six"][c])]
      for n, c in [("source test", "source_test"), ("cross-participant", "target")]])
a4 = lambda d: [g3(d["density"]), g3(d["confidence"]), g3(d["energy"])]
rows = [["Held-out gesture"] + a4(N["novelty"]) + ["n/a"]]
for f in ["channel rotation", "gain shift", "channel dropout", "broadband noise", "motion artefact"]:
    rows.append([f.replace("artefact", "artifact").capitalize()] + a4(S[f]["auroc"]) + [g3(S[f]["auroc"]["range check"])])
for c, n in [("cross-participant", "New participant"), ("cross-dataset", "Cross-dataset"), ("mismatch", "Conditioning mismatch")]:
    rows.append([n] + a4(L[c]["auroc"]) + [g3(L[c]["auroc"]["range check"])])
save("table04_detection", ["Condition", "Density", "Max softmax", "Energy", "Range check"], rows)
rb = lambda tr, te, d: [tr, te, f3(d["bal"]), g3(d["conf"])] + ([g3(d["auroc"][k]) for k in ["confidence", "density", "range check"]] if "auroc" in d else ["ref."] * 3)
save("table05_within_B", ["Trained on", "Tested on", "Balanced acc.", "Mean conf.", "Max softmax", "Density", "Range check"],
     [rb("rectified", "rectified", RB["matched"]), rb("rectified", "signed", RB["signed (no rectification)"]),
      rb("signed", "signed", SB["matched"]), rb("signed", "rectified", SB["rectified"]),
      rb("both", "rectified", XB["matched"]), rb("both", "signed", XB["signed (no rectification)"]), rb("both", "gain x0.1", XB["gain x0.1"])])
fam = lambda d: [f3(d["bal"]), g3(d["auroc"]["confidence"]), g3(d["auroc"]["density"]), g3(d["auroc"]["range check"])]
save("table06_mismatch_family", ["Dataset", "Mismatch", "Balanced acc.", "Max softmax", "Density", "Range check"],
     [["A", c] + fam(S[c]) for c in ["gain x0.1", "gain x0.25", "gain x0.5", "gain x2", "gain x4", "gain x10", "envelope 150 ms",
                                       "per-window z-score", "400 Hz stream"]] +
     [["B", c] + fam(RB[c]) for c in ["gain x0.1", "gain x4", "envelope 150 ms", "per-window z-score", "400 Hz stream", "band-pass 20-90 Hz"]])
tests = {t["name"]: t for t in N["tests"]}
save("fig6_error_ranking_values", ["Statistic", "AURC", "Bootstrap difference vs max softmax", "BH q"],
     [[k, f"{N['aurc'][k][0]:.4f} ± {N['aurc'][k][1]:.4f}", "" if k == "confidence" else f"{N['boot'][k]['d']:.4f} [{N['boot'][k]['lo']:.4f}, {N['boot'][k]['hi']:.4f}]",
       "" if k == "confidence" else f"{tests[k.capitalize() + ' vs max softmax, error ranking']['q']:.4f}"] for k in ["confidence", "entropy", "margin", "density", "energy"]])
lr = N["lda_ladder3"]; lw, lwm = RB["matched"], RB["signed (no rectification)"]
save("table07_classical_lda", ["Condition", "Learned bal.", "LDA bal.", "Learned max softmax AUROC", "LDA max softmax AUROC", "LDA density AUROC"],
     [["A source test (6)", g3(N["six"]["source_test"]["bal"]), g3(N["lda_source_test_bal6"]), "ref.", "ref.", "ref."]] +
     [[f"A {c} (6)", g3(S[c]["bal"]), g3(S[c]["lda_bal"]), g3(S[c]["auroc"]["confidence"]), g3(S[c]["lda_auroc"]["confidence"]), g3(S[c]["lda_auroc"]["density"])]
      for c in ["gain x0.1", "envelope 150 ms"]] +
     [[f"A to B {n} (3)", g3(L[c]["bal"]), g3(lr[c]["bal"]), g3(L[c]["auroc"]["confidence"]), g3(lr[c]["auroc"]["confidence"]), g3(lr[c]["auroc"]["density"])]
      for n, c in [("rectified", "cross-dataset"), ("signed", "mismatch")]] +
     [["Within B matched (3)", g3(lw["bal"]), g3(lw["lda_bal"]), "ref.", "ref.", "ref."],
      ["Within B signed (3)", g3(lwm["bal"]), g3(lwm["lda_bal"]), g3(lwm["auroc"]["confidence"]), g3(lwm["lda_auroc"]["confidence"]), g3(lwm["lda_auroc"]["density"])]])
save("table08_layered_checks", ["Gate", "Clean accepted %", "Faulty accepted %", "Balanced acc. accepted", "Harmful %"],
     [[k, f"{100 * v['cov_clean'][0]:.1f}", f"{100 * v['cov_faulty'][0]:.1f}", g3(v["bal_accepted"]), f"{100 * v['harmful'][0]:.1f} ± {100 * v['harmful'][1]:.1f}"] for k, v in C.items()])
save("table09_depth", ["Tap", "Balanced acc.", "Mean fault detection AUROC", "AURC"],
     [[t, f3(D[t]["bal"]), f"{D[t]['ood'][0]:.4f} ± {D[t]['ood'][1]:.4f}", f"{D[t]['aurc'][0]:.4f} ± {D[t]['aurc'][1]:.4f}"] for t in D])
cls = ["rest", "fist", "open palm", "wrist left", "wrist right", "thumbs-up"]
(rs, ss), (rt, st) = N["recall6"]["source_test"], N["recall6"]["target"]
save("table10_per_class", ["Class", "Recall source test", "Recall cross-participant", "Drop"],
     [[cls[i], f"{rs[i]:.3f} ± {ss[i]:.3f}", f"{rt[i]:.3f} ± {st[i]:.3f}", f"{rs[i] - rt[i]:.3f}"] for i in np.argsort(np.array(rs) - np.array(rt))])
pp = pickle.load(open(os.path.join(RES, "v3_pp.pkl"), "rb"))
save("table11_per_participant", ["Participant", "Accuracy", "Balanced acc.", "Mean conf."],
     [[j, f3(v["acc"]), f3(v["bal"]), f3(v["conf"])] for j, v in sorted(pp.items())])
save("table12_tests", ["Test", "p", "Bonferroni", "BH q", "Survives"],
     [[t["name"], f"{t['p']:.2e}", f"{t['bonf']:.2e}", f"{t['q']:.2e}", t["q"] < 0.05] for t in N["tests"]])
MM = L["mismatch"]
print("Headline numbers (values reported in the paper in brackets; a QUICK run will not match):")
print(f"  mismatch balanced acc.            {f3(MM['bal'])}   [0.342 ± 0.015]")
print(f"  mismatch max-softmax AUROC        {f3(MM['auroc']['confidence'])}   [0.179 ± 0.094]")
print(f"  mismatch density AUROC            {f3(MM['auroc']['density'])}   [0.671 ± 0.013]")
print(f"  within B, rect->signed MSP AUROC  {f3(RB['signed (no rectification)']['auroc']['confidence'])}   [0.031 ± 0.022]")
print(f"  gain x0.1 MSP / density AUROC     {g3(S['gain x0.1']['auroc']['confidence'])} / {g3(S['gain x0.1']['auroc']['density'])}   [0.174 / 0.378]")
print(f"  harmful acceptance, none -> all   {100 * C['none']['harmful'][0]:.1f}% -> {100 * C['range + density + confidence']['harmful'][0]:.1f}%   [22.9% -> 3.3%]")
print(f"  tables written to {OUT}")

from core import X1, y1, s1, ss1, X2, y2, s2, IDX
rows = [[k, len(ix), "/".join(map(str,np.bincount(y1[ix],minlength=6)))] for k,ix in IDX.items()]
rows += [["Dataset B",len(y2),"/".join(map(str,np.bincount(y2,minlength=3)))]]
save("table02_counts", ["Partition","Windows","Class counts"], rows)
save("audit_intervals", ["Metric","Lower95","Upper95"], [[k,*v] for k,v in N["participant_ci"].items()])
save("audit_aurc_ties", ["Statistic","Legacy AURC","Tie neutral AURC"], [[k,v["mean_legacy"],v["mean_neutral"]] for k,v in N["tie_sensitivity"].items()])
