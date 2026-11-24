"""Figures for the revised manuscript, single-column width, from N.pkl."""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
RES = os.environ.get("EMG_RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
N = pickle.load(open(os.path.join(RES, "N.pkl"), "rb"))
OUT = os.path.join(RES, "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
CD, CC, CN, CB, CO = "#b02418", "#1a7a4c", "#6b6b6b", "#2a5d9c", "#c98a00"
W = 3.4

# ---- Fig. 2: ladder ------------------------------------------------------------
L = N["ladder3"]; conds = ["source_test", "cross-participant", "cross-dataset", "mismatch"]
lbl = ["source\ntest", "cross-\nparticipant", "cross-\ndataset", "conditioning\nmismatch"]
x = np.arange(4)
fig, ax = plt.subplots(3, 1, figsize=(W, 4.6))
ax[0].errorbar(x, [L[c]["acc"][0] for c in conds], [L[c]["acc"][1] for c in conds], marker="o", ms=3.5, ls="--", color=CN, capsize=2, label="pooled accuracy")
ax[0].errorbar(x, [L[c]["bal"][0] for c in conds], [L[c]["bal"][1] for c in conds], marker="s", ms=3.5, color=CD, capsize=2, label="balanced accuracy")
ax[0].axhline(1/3, color="k", ls=":", lw=0.7); ax[0].set_ylabel("accuracy"); ax[0].set_title("(a) pooled accuracy conceals the collapse")
ax[0].legend(loc="lower left", frameon=False)
ax[1].errorbar(x, [L[c]["conf"][0] for c in conds], [L[c]["conf"][1] for c in conds], marker="o", ms=3.5, color=CC, capsize=2, label="mean confidence")
ax[1].errorbar(x, [L[c]["bal"][0] for c in conds], [L[c]["bal"][1] for c in conds], marker="s", ms=3.5, color=CD, capsize=2, label="balanced accuracy")
ax[1].set_ylabel("value"); ax[1].set_title("(b) confidence rises as accuracy falls"); ax[1].legend(loc="lower left", frameon=False)
ax[2].bar(x, [L[c]["refusal"][0] * 100 for c in conds], 0.6, yerr=[L[c]["refusal"][1] * 100 for c in conds], color="#8fa9c9", edgecolor="#333", lw=0.4, capsize=2)
ax[2].set_ylabel("refusal rate (%)"); ax[2].set_title("(c) refusal across evaluation conditions")
for a in ax: a.set_xticks(x); a.set_xticklabels(lbl, fontsize=5.8); a.grid(alpha=0.25, axis="y")
fig.tight_layout(h_pad=0.8); fig.savefig(f"{OUT}/fig2.png"); plt.close(fig)

# ---- Fig. 3: inversion across runs ------------------------------------------------
B = N["B"]
groups = [("A to B, 10 seeds", None), ("A to B, 10 splits", N["splits"]["mismatch"]["auroc_raw"]),
          ("within B, rectified\nto signed", B["rectified"]["signed (no rectification)"]["auroc_raw"]),
          ("within B, signed\nto rectified", B["signed"]["rectified"]["auroc_raw"])]
seedv = N["mismatch_seed_raw"]
groups[0] = (groups[0][0], seedv)
fig, ax = plt.subplots(figsize=(W, 2.3)); rng = np.random.RandomState(0)
for i, (nm, d) in enumerate(groups):
    y = len(groups) - 1 - i
    for k, col, off, mk in [("confidence", CC, 0.14, "o"), ("density", CD, -0.14, "s")]:
        v = np.array(d[k]); ax.scatter(v, y + off + rng.uniform(-0.05, 0.05, len(v)), s=9, color=col, marker=mk,
                                       edgecolor="#222", lw=0.3, label=("max softmax" if k == "confidence" else "density") if i == 0 else None, zorder=3)
        ax.plot([v.mean()] * 2, [y + off - 0.1, y + off + 0.1], color=col, lw=1.4, zorder=4)
ax.axvline(0.5, color="k", ls="--", lw=0.7); ax.set_xlim(0, 1)
ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups][::-1], fontsize=6)
ax.set_xlabel("detection AUROC (mismatched vs. matched input)"); ax.grid(alpha=0.25, axis="x")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=5.8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3.png"); plt.close(fig)

# ---- Fig. 4: mismatch family on A ------------------------------------------------
S = N["shift"]
fam = ["gain x0.1", "gain x0.25", "gain x0.5", "gain x2", "gain x4", "gain x10", "envelope 50 ms", "envelope 150 ms",
       "per-window z-score", "400 Hz stream"]
nice = ["gain ×0.1", "gain ×0.25", "gain ×0.5", "gain ×2", "gain ×4", "gain ×10", "envelope 50 ms", "envelope 150 ms",
        "per-window std.", "400 Hz stream"]
y = np.arange(len(fam))[::-1]
fig, ax = plt.subplots(2, 1, figsize=(W, 4.4), gridspec_kw=dict(height_ratios=[1, 1.25]))
ax[0].barh(y, [S[c]["bal"][0] for c in fam], 0.6, xerr=[S[c]["bal"][1] for c in fam], color="#8fa9c9", edgecolor="#333", lw=0.4, capsize=1.5)
ax[0].axvline(S and N["six"]["source_test"]["bal"][0], color=CB, ls="-", lw=0.8)
ax[0].axvline(1/6, color="k", ls=":", lw=0.7)
ax[0].set_yticks(y); ax[0].set_yticklabels(nice, fontsize=5.8); ax[0].set_xlim(0, 1)
ax[0].set_xlabel("balanced accuracy (six classes)"); ax[0].set_title("(a) accuracy under each mismatch")
h = 0.26
for k, col, off, nm in [("confidence", CC, h, "max softmax"), ("density", CD, 0, "density"), ("range check", CN, -h, "range check")]:
    ax[1].barh(y + off, [S[c]["auroc"][k][0] for c in fam], h, color=col, edgecolor="#333", lw=0.3, label=nm)
ax[1].axvline(0.5, color="k", ls="--", lw=0.7); ax[1].set_xlim(0, 1)
ax[1].set_yticks(y); ax[1].set_yticklabels(nice, fontsize=5.8); ax[1].set_xlabel("detection AUROC")
ax[1].set_title("(b) detection of each mismatch"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, frameon=False, fontsize=5.6)
for a in ax: a.grid(alpha=0.25, axis="x")
fig.tight_layout(h_pad=0.8); fig.savefig(f"{OUT}/fig4.png"); plt.close(fig)

# ---- Fig. 5: fault suite detection ----------------------------------------------
rows = [("held-out gesture", N["novelty"])] + [(n.replace("artefact", "artifact"), S[n]["auroc"]) for n in
        ["channel rotation", "gain shift", "channel dropout", "broadband noise", "motion artefact"]] + \
       [("new participant", L["cross-participant"]["auroc"]), ("cross-dataset", L["cross-dataset"]["auroc"]),
        ("conditioning mismatch", L["mismatch"]["auroc"])]
y = np.arange(len(rows))[::-1]; h = 0.36
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.barh(y + h/2, [r[1]["density"][0] for r in rows], h, xerr=[r[1]["density"][1] for r in rows], color=CD, edgecolor="#333", lw=0.35, capsize=1.2, label="density")
ax.barh(y - h/2, [r[1]["confidence"][0] for r in rows], h, xerr=[r[1]["confidence"][1] for r in rows], color=CC, edgecolor="#333", lw=0.35, capsize=1.2, label="max softmax")
ax.axvline(0.5, color="k", ls="--", lw=0.8); ax.axhspan(-0.5, 2.5, color="#f2f2f2", zorder=0)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6); ax.set_xlim(0, 1)
ax.set_xlabel("detection AUROC"); ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=5.8); ax.grid(alpha=0.25, axis="x")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5.png"); plt.close(fig)

# ---- Fig. 6: error ranking -------------------------------------------------------
st = ["confidence", "entropy", "margin", "density", "energy"]; nm = ["max softmax", "entropy", "margin", "density", "energy"]
fig, ax = plt.subplots(2, 1, figsize=(W, 3.6))
ax[0].bar(range(5), [N["aurc"][k][0] for k in st], 0.6, yerr=[N["aurc"][k][1] for k in st], capsize=1.8,
          color=[CC, CC, CC, CD, CO], edgecolor="#333", lw=0.35)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(nm, fontsize=6); ax[0].set_ylabel("AURC (lower is better)")
ax[0].set_title("(a) error ranking across participants"); ax[0].grid(alpha=0.25, axis="y")
bk = ["entropy", "margin", "density", "energy"]; bo = N["boot"]
ax[1].errorbar([bo[k]["d"] for k in bk], range(4)[::-1], xerr=[[bo[k]["d"] - bo[k]["lo"] for k in bk], [bo[k]["hi"] - bo[k]["d"] for k in bk]],
               fmt="o", ms=3.5, color="#333", capsize=2)
ax[1].axvline(0, color="k", ls="--", lw=0.7); ax[1].set_yticks(range(4)[::-1]); ax[1].set_yticklabels(bk, fontsize=6)
ax[1].set_xlabel("difference in AURC vs. max softmax (95% interval)"); ax[1].set_title("(b) participant-clustered bootstrap")
ax[1].grid(alpha=0.25, axis="x")
fig.tight_layout(h_pad=0.8); fig.savefig(f"{OUT}/fig6.png"); plt.close(fig)

# ---- Fig. 7: cascade -------------------------------------------------------------
C = N["cascade"]; pol = ["none", "confidence", "density", "range check", "range + density", "range + density + confidence"]
pn = ["no gate", "max\nsoftmax", "density", "range\ncheck", "range +\ndensity", "all three"]
fig, ax = plt.subplots(2, 1, figsize=(W, 3.7))
x = np.arange(len(pol))
ax[0].bar(x, [C[p]["harmful"][0] * 100 for p in pol], 0.6, yerr=[C[p]["harmful"][1] * 100 for p in pol], color=CD, edgecolor="#333", lw=0.35, capsize=1.8)
ax[0].set_ylabel("harmful acceptance (%)"); ax[0].set_title("(a) faulty windows accepted and misclassified")
w = 0.38
ax[1].bar(x - w/2, [C[p]["cov_clean"][0] * 100 for p in pol], w, color=CB, edgecolor="#333", lw=0.35, label="clean windows")
ax[1].bar(x + w/2, [C[p]["cov_faulty"][0] * 100 for p in pol], w, color="#d9a29a", edgecolor="#333", lw=0.35, label="faulty windows")
ax[1].set_ylabel("accepted (%)"); ax[1].set_title("(b) coverage"); ax[1].legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=5.6)
for a in ax: a.set_xticks(x); a.set_xticklabels(pn, fontsize=5.4); a.grid(alpha=0.25, axis="y")
fig.tight_layout(h_pad=0.8); fig.savefig(f"{OUT}/fig7.png"); plt.close(fig)

# ---- Fig. 8: 200 ms and factorial ------------------------------------------------
w2 = N["w200"]; x = np.arange(4)
fig, ax = plt.subplots(3, 1, figsize=(W, 4.8))
ax[0].errorbar(x, [w2[c]["bal"][0] for c in conds], [w2[c]["bal"][1] for c in conds], marker="s", ms=3.5, color=CD, capsize=2, label="balanced accuracy")
ax[0].errorbar(x, [w2[c]["conf"][0] for c in conds], [w2[c]["conf"][1] for c in conds], marker="o", ms=3.5, color=CC, capsize=2, label="mean confidence")
ax[0].axhline(1/3, color="k", ls=":", lw=0.7); ax[0].set_xticks(x); ax[0].set_xticklabels(lbl, fontsize=5.8)
ax[0].set_title("(a) shift ladder with 200 ms windows"); ax[0].legend(loc="lower left", frameon=False)
c3 = ["cross-participant", "cross-dataset", "mismatch"]; x3 = np.arange(3); h = 0.36
ax[1].bar(x3 - h/2, [w2[c]["auroc"]["density"][0] for c in c3], h, yerr=[w2[c]["auroc"]["density"][1] for c in c3], color=CD, edgecolor="#333", lw=0.35, capsize=1.5, label="density")
ax[1].bar(x3 + h/2, [w2[c]["auroc"]["confidence"][0] for c in c3], h, yerr=[w2[c]["auroc"]["confidence"][1] for c in c3], color=CC, edgecolor="#333", lw=0.35, capsize=1.5, label="max softmax")
ax[1].axhline(0.5, color="k", ls="--", lw=0.7); ax[1].set_xticks(x3); ax[1].set_xticklabels(["cross-participant", "cross-dataset", "mismatch"], fontsize=5.8)
ax[1].set_ylim(0, 1); ax[1].set_ylabel("detection AUROC"); ax[1].set_title("(b) detection with 200 ms windows"); ax[1].legend(loc="upper left", frameon=False)
fc = ["cross-dataset", "mismatch", "signed + own norm", "rectified + own norm"]
fl = ["rectified,\nsource stats", "signed,\nsource stats", "signed,\nown stats", "rectified,\nown stats"]
x4 = np.arange(4)
ax[2].bar(x4 - 0.27, [L[c]["bal"][0] for c in fc], 0.27, color="#8fa9c9", edgecolor="#333", lw=0.35, label="balanced acc.")
ax[2].bar(x4, [L[c]["auroc"]["confidence"][0] for c in fc], 0.27, color=CC, edgecolor="#333", lw=0.35, label="max softmax AUROC")
ax[2].bar(x4 + 0.27, [L[c]["auroc"]["density"][0] for c in fc], 0.27, color=CD, edgecolor="#333", lw=0.35, label="density AUROC")
ax[2].axhline(0.5, color="k", ls="--", lw=0.7); ax[2].set_xticks(x4); ax[2].set_xticklabels(fl, fontsize=5.6); ax[2].set_ylim(0, 1.05)
ax[2].set_title("(c) conditioning by normalization, Dataset B input"); ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False, fontsize=5.4, ncol=3)
for a in ax: a.grid(alpha=0.25, axis="y")
fig.tight_layout(h_pad=0.8); fig.savefig(f"{OUT}/fig8.png"); plt.close(fig)

# ---- Fig. 9 and 10: depth ----------------------------------------------------------
D = N["depth"]; taps = ["proj", "layer1", "layer2", "layer3_norm"]; tl = ["patch\nprojection", "block 1", "block 2", "block 3\n(output)"]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(W, 2.3))
l1, = ax.plot(x, [D[t]["ood"][0] for t in taps], "-o", ms=3.5, color=CD, label="mean fault detection AUROC")
l2, = ax.plot(x, [D[t]["bal"][0] for t in taps], "-s", ms=3.5, color=CB, label="balanced accuracy")
ax.set_ylim(0.55, 0.75); ax.set_ylabel("AUROC / balanced accuracy"); ax.set_xticks(x); ax.set_xticklabels(tl, fontsize=6)
ax.grid(alpha=0.25, axis="y")
ax2 = ax.twinx()
l3, = ax2.plot(x, [D[t]["aurc"][0] for t in taps], "--^", ms=3.5, color=CC, label="AURC of max softmax (right axis)")
ax2.set_ylim(0.050, 0.065); ax2.set_ylabel("AURC (lower is better)"); ax2.tick_params(labelsize=6)
ax.legend(handles=[l1, l2, l3], loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig9.png"); plt.close(fig)
faults = list(D["proj"]["fault"]); fn = [f.replace("artefact", "artifact") for f in faults]
fig, ax = plt.subplots(figsize=(W, 2.4)); cols = [CD, CB, CC, CO, CN]
for f, n_, c in zip(faults, fn, cols):
    ax.errorbar(x, [D[tp]["fault"][f][0] for tp in taps], [D[tp]["fault"][f][1] for tp in taps], marker="o", ms=3, color=c, capsize=1.5, label=n_)
ax.axhline(0.5, color="k", ls="--", lw=0.7); ax.set_xticks(x); ax.set_xticklabels(tl, fontsize=5.8)
ax.set_ylabel("density detection AUROC"); ax.grid(alpha=0.25, axis="y")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=5.6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig10.png"); plt.close(fig)
print("figures written")

# Fig. 1: explicit branching; the two B conditions use the same recordings.
from matplotlib.patches import FancyBboxPatch
from core import X2
fig, ax = plt.subplots(figsize=(3.4, 2.7)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
def box(x,y,w,h,title,detail,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006',facecolor=color,edgecolor='#333',lw=.6))
    ax.text(x+w/2,y+h*.68,title,ha='center',va='center',fontsize=7,fontweight='bold')
    ax.text(x+w/2,y+h*.28,detail,ha='center',va='center',fontsize=6)
def arrow(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color='#333',lw=.7))
box(.12,.79,.76,.16,'Source training','A: participants 1–20, sessions 1–4; n = 1600','#dfe8f5')
box(.12,.56,.76,.16,'Held-out session','A: participants 1–20, session 5; n = 400','#dfe8f5')
box(.12,.33,.76,.16,'New participants','A: participants 21–30; n = 1000','#f5ead7')
box(.015,.05,.455,.17,'B rectified',f'Same B recordings; n = {len(X2)}','#f7ddd7')
box(.53,.05,.455,.17,'B signed',f'Same B recordings; n = {len(X2)}','#efcac2')
arrow(.5,.79,.5,.72);arrow(.5,.56,.5,.49);arrow(.5,.33,.24,.22);arrow(.5,.33,.76,.22)
fig.tight_layout(pad=.3);fig.savefig(f'{OUT}/fig1.png');plt.close(fig)
