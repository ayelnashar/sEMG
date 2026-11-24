"""Shared helpers for the revision experiments: deployment conditions, classical
baseline, probe-only statistics, detection metrics."""
import numpy as np, torch
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from core import stats_all, Gate

FS = 200

# ---------------------------------------------------------------- conditions
# Each deployment condition maps a RAW window batch (B,1000,8) to the input the
# model receives, given the training conditioning `conv` and the training
# normalization (mu, sd). The model always expects norm(conv(x)).
def _bp(X, lo=20.0, hi=90.0):
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, X, axis=1).astype(np.float32)

def _up2(X):
    Y = resample_poly(X, 2, 1, axis=1).astype(np.float32)
    n = X.shape[1]; s = (Y.shape[1] - n) // 2
    return Y[:, s:s + n]

def make_conditions(conv, mu, sd, signed_raw_available, rect_train):
    """Return dict name -> f(Xraw) giving the model input. `conv` is the training
    conditioning applied to raw data (abs for rectified, identity for signed)."""
    N = lambda X: ((X - mu) / sd).astype(np.float32)
    C = {}
    C["matched"] = lambda X: N(conv(X))
    for g in [0.1, 0.25, 0.5, 2.0, 4.0, 10.0]:
        C[f"gain x{g:g}"] = (lambda g: lambda X: N(conv(X) * g))(g)
    C["envelope 50 ms"] = lambda X: N(uniform_filter1d(conv(X), 10, axis=1))
    C["envelope 150 ms"] = lambda X: N(uniform_filter1d(conv(X), 30, axis=1))
    def perwin(X):
        Y = conv(X); m = Y.mean(1, keepdims=True); s = Y.std(1, keepdims=True) + 1e-6
        return ((Y - m) / s).astype(np.float32)
    C["per-window z-score"] = perwin
    if rect_train:
        C["400 Hz stream"] = lambda X: N(np.maximum(_up2(conv(X)), 0))
    else:
        C["400 Hz stream"] = lambda X: N(_up2(conv(X)))
    if signed_raw_available:
        C["band-pass 20-90 Hz"] = lambda X: N(conv(_bp(X)))
        if rect_train:
            C["signed (no rectification)"] = lambda X: N(X)
        else:
            C["rectified"] = lambda X: N(np.abs(X))
    return C

# ---------------------------------------------------------------- probe-only
def probe_stats(base, Z):
    with torch.no_grad():
        lg = base(torch.from_numpy(Z)).numpy()
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return p, -p.max(1)

def all_stats(env, gate, Z, base):
    p, sc = stats_all(env, gate, Z, base)
    pp, cp = probe_stats(base, Z)
    sc["confidence (probe only)"] = cp
    return p, pp, sc

def auroc(s_in, s_out):
    lab = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
    return float(roc_auc_score(lab, np.r_[s_in, s_out]))

def tpr_at(s_in, s_out, fr=0.05):
    thr = np.percentile(s_in, 100 * (1 - fr))
    return float((s_out > thr).mean())

def contract_score(Xin, mu, sd, lo, hi):
    """Fraction of samples of the model input, mapped back to raw units, that fall
    outside the range seen in (conditioned) training data."""
    R = Xin * sd + mu
    return ((R < lo - 1e-3) | (R > hi + 1e-3)).mean(axis=(1, 2))

# ---------------------------------------------------------------- classical
def td_features(X, thr=1.0):
    """Hudgins time-domain set per channel on the model input (raw units)."""
    mav = np.abs(X).mean(1)
    wl = np.abs(np.diff(X, axis=1)).sum(1) / X.shape[1]
    a, b = X[:, :-1], X[:, 1:]
    zc = (((a * b) < 0) & (np.abs(a - b) >= thr)).mean(1)
    d1, d2 = X[:, 1:-1] - X[:, :-2], X[:, 1:-1] - X[:, 2:]
    ssc = (((d1 * d2) > 0) & ((np.abs(d1) >= thr) | (np.abs(d2) >= thr))).mean(1)
    return np.concatenate([np.log(mav + 1e-3), np.log(wl + 1e-3), zc, ssc], 1).astype(np.float64)

class LDABaseline:
    def fit(self, F, y, k):
        self.m, self.s = F.mean(0), F.std(0)
        self.s[self.s < 1e-8] = 1.0
        Fz = (F - self.m) / self.s
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Fz, y)
        self.gate = Gate().fit(Fz, y, k)
        return self
    def run(self, F):
        Fz = (F - self.m) / self.s
        p = self.lda.predict_proba(Fz)
        return p, dict(confidence=-p.max(1), density=self.gate.score(Fz))
