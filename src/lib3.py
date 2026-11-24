"""Probe-only classifier and statistics for the revised paper (no residual head)."""
import numpy as np, torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from core import Baseline, Gate, ece, aurc

def fit_probe(Zf, yf, Zc, yc, k, seed=42):
    """Multinomial linear probe (AdamW, 300 full-batch steps) followed by
    temperature scaling on the calibration participants (L-BFGS)."""
    if set(np.unique(yf)) != set(range(k)) or not np.isfinite(Zf).all():
        raise ValueError("Probe fitting requires finite features and every training class")
    torch.manual_seed(seed)
    base = Baseline(Zf.shape[1], k)
    Z, Y = torch.from_numpy(Zf), torch.from_numpy(yf)
    o = torch.optim.AdamW(base.lin.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(300):
        o.zero_grad(); Fn.cross_entropy(base.raw(Z), Y).backward(); o.step()
    with torch.no_grad(): lc = base.raw(torch.from_numpy(Zc))
    lt = torch.zeros(1, requires_grad=True)
    o2 = torch.optim.LBFGS([lt], lr=0.1, max_iter=100)
    def cl():
        o2.zero_grad(); l = Fn.cross_entropy(lc / lt.exp(), torch.from_numpy(yc)); l.backward(); return l
    o2.step(cl)
    with torch.no_grad(): base.logT.copy_(lt.detach())
    return base

STATS = ["density", "confidence", "entropy", "margin", "energy"]

def pstats(base, gate, Z):
    with torch.no_grad():
        raw = base.raw(torch.from_numpy(Z)).numpy()
        lg = base(torch.from_numpy(Z)).numpy()          # calibrated logits
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    part = np.partition(p, -2, 1)
    m = raw.max(1, keepdims=True)
    energy = -(m[:, 0] + np.log(np.exp(raw - m).sum(1)))   # negative log-sum-exp of probe logits
    return p, dict(density=gate.score(Z), confidence=-p.max(1),
                   entropy=-(p * np.log(p + 1e-12)).sum(1),
                   margin=-(part[:, -1] - part[:, -2]), energy=energy)

def auroc(s_in, s_out):
    lab = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
    return float(roc_auc_score(lab, np.r_[s_in, s_out]))

def tpr_at(s_in, s_out, fr=0.05):
    return float((s_out > np.percentile(s_in, 100 * (1 - fr))).mean())

def macro_auc(p, y, k):
    pres = [c for c in range(k) if (y == c).any() and (y != c).any()]
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, c]) for c in pres]))

def summary(p, y, k, gate, Z):
    pr = p.argmax(1); ref = gate.refuse(Z)
    out = dict(acc=float((pr == y).mean()), bal=float(balanced_accuracy_score(y, pr)),
               mauc=macro_auc(p, y, k), ece=ece(p, y), conf=float(p.max(1).mean()),
               refusal=float(ref.mean()),
               recall=[float((pr[y == c] == c).mean()) if (y == c).any() else np.nan for c in range(k)],
               pred_hist=np.bincount(pr, minlength=k).tolist())
    if ref.sum() > 5 and (~ref).sum() > 5:
        out["bal_accepted"] = float(balanced_accuracy_score(y[~ref], pr[~ref]))
        out["bal_refused"] = float(balanced_accuracy_score(y[ref], pr[ref]))
        out["err_accepted"] = float((pr[~ref] != y[~ref]).mean())
        out["err_refused"] = float((pr[ref] != y[ref]).mean())
    return out
