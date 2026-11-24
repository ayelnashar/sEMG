import os, sys, time, json, math, glob, re, warnings
import numpy as np, torch
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
if DEV == "cpu": torch.set_num_threads(max(1, int(os.environ.get("EMG_THREADS", 2))))
DATA = os.environ.get("EMG_DATA", os.path.dirname(os.path.abspath(__file__)))
EPOCHS=int(os.environ.get("EPOCHS",25)); BATCH=64
PATCH, N_TOK, D_MODEL, N_LAYER, N_HEAD = 20, 50, 96, 3, 4
PATCH_S, N_TOK_S = 5, 8
MASK_RATIO=0.40; DELTA=1.5; COV_RIDGE=1e-2; OOD_PCT=95.0
CALIB_SUBJECTS=[17,18,19,20]
N_T, N_CH, K6, KC = 1000, 8, 6, 3
D1_MAP = {0: 0, 1: 1, 3: 2, 4: 2}
def seed_all(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def norm_fit(X):
    return X.mean((0,1), keepdims=True), X.std((0,1), keepdims=True) + 1e-6
# ===== CELL 8 =====
import torch.nn as nn

class EMGEncoder(nn.Module):
    def __init__(self, d=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, n_ch=N_CH,
                 n_tok=N_TOK, patch=PATCH):
        super().__init__()
        self.n_tok = n_tok
        self.pd = patch * n_ch
        self.proj = nn.Linear(self.pd, d)
        self.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)
        self.mask_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_head, 4*d, dropout=0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, self.pd)
    def patchify(self, x): return x.reshape(x.shape[0], self.n_tok, self.pd)
    def forward(self, x, mask=None):
        p = self.patchify(x); h = self.proj(p)
        if mask is not None: h = torch.where(mask.unsqueeze(-1), self.mask_tok.expand_as(h), h)
        return self.norm(self.enc(h + self.pos)), p
    @torch.no_grad()
    def taps(self, x):
        # representations at four depths, for the depth analysis
        h = self.proj(self.patchify(x)) + self.pos
        out = {"proj": h.mean(1)}
        z = h
        for i, layer in enumerate(self.enc.layers, 1):
            z = layer(z)
            if i < len(self.enc.layers): out[f"layer{i}"] = z.mean(1)
        out["layer3_norm"] = self.norm(z).mean(1)
        return out

def pretrain(Xtr, epochs=EPOCHS, seed=42, bs=BATCH, lr=3e-4, verbose=False,
             n_tok=N_TOK, patch=PATCH):
    seed_all(seed)
    m = EMGEncoder(n_tok=n_tok, patch=patch).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X = torch.from_numpy(Xtr).to(DEV); n = len(X)
    nm = max(1, int(round(MASK_RATIO * n_tok)))
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEV); tot = cnt = 0.0
        for i in range(0, n, bs):
            xb = X[perm[i:i+bs]]; B = len(xb)
            r = torch.rand(B, n_tok, device=DEV)
            mask = torch.zeros(B, n_tok, dtype=torch.bool, device=DEV)
            mask.scatter_(1, r.argsort(1)[:, :nm], True)
            h, tgt = m(xb, mask); loss = ((m.head(h) - tgt) ** 2)[mask].mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += loss.item()*B; cnt += B
        sch.step()
        if verbose and ep % 5 == 0: print(f"  ep {ep:3d} masked-MSE {tot/cnt:.4f}")
    return m, tot/cnt

@torch.no_grad()
def embed(m, X, bs=256, tap=None):
    m.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(DEV)
        out.append((m.taps(xb)[tap] if tap else m(xb)[0].mean(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

#print("encoder parameters, 5 s config:", sum(p.numel() for p in EMGEncoder().parameters()))
#print("encoder parameters, 200 ms config:", sum(p.numel() for p in EMGEncoder(n_tok=N_TOK_S, patch=PATCH_S).parameters()))


# ===== CELL 10 =====
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import beta as Beta, ttest_rel, spearmanr, t as tdist

class Baseline(nn.Module):
    def __init__(s, d, k):
        super().__init__(); s.lin = nn.Linear(d, k)
        s.logT = nn.Parameter(torch.zeros(1), requires_grad=False)
    def raw(s, z): return s.lin(z)
    def forward(s, z): return s.raw(z) / s.logT.exp()

class EnvHead(nn.Module):
    def __init__(s, d, k, delta=DELTA):
        super().__init__(); s.delta = delta
        s.net = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,k))
        nn.init.zeros_(s.net[3].weight); nn.init.zeros_(s.net[3].bias)
    def corr(s, z): return s.net(z).clamp(-s.delta, s.delta)

class Envelope(nn.Module):
    def __init__(s, base, head): super().__init__(); s.base, s.head = base, head
    def forward(s, z):
        with torch.no_grad(): b = s.base(z)
        return b + s.head.corr(z), b

def fit_envelope(Zf, yf, Zc, yc, k, seed=42, delta=DELTA):
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
        o2.zero_grad(); l = Fn.cross_entropy(lc/lt.exp(), torch.from_numpy(yc)); l.backward(); return l
    o2.step(cl)
    with torch.no_grad(): base.logT.copy_(lt.detach())
    head = EnvHead(Zf.shape[1], k, delta); env = Envelope(base, head)
    o3 = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(400):
        o3.zero_grad(); f, _ = env(Z); Fn.cross_entropy(f, Y).backward(); o3.step()
    return base, env, float(lt.exp())

class Gate:
    def fit(s, Z, y, k, ridge=COV_RIDGE, pct=OOD_PCT):
        if not np.isfinite(Z).all() or set(np.unique(y)) != set(range(k)):
            raise ValueError("Density fitting requires finite features and every training class")
        s.mu = np.stack([Z[y==c].mean(0) for c in range(k)])
        C = np.zeros((Z.shape[1],)*2)
        for c in range(k):
            d = Z[y==c] - s.mu[c]; C += d.T @ d
        C /= len(Z); C += ridge*np.trace(C)/C.shape[0]*np.eye(C.shape[0])
        s.P = np.linalg.inv(C); s.thr = np.percentile(s.score(Z), pct); return s
    def score(s, Z): return np.stack([((Z-m)@s.P*(Z-m)).sum(1) for m in s.mu]).min(0)
    def refuse(s, Z): return s.score(Z) > s.thr

def stats_all(env, gate, Z, base):
    with torch.no_grad(): f, _ = env(torch.from_numpy(Z))
    lg = f.numpy(); p = torch.softmax(f,1).numpy()
    part = np.partition(p, -2, 1); T = float(base.logT.exp())
    torch.manual_seed(0); env.head.train(); P = []
    with torch.no_grad():
        b = base(torch.from_numpy(Z))
        for _ in range(30): P.append(torch.softmax(b + env.head.corr(torch.from_numpy(Z)),1).numpy())
    env.head.eval(); P = np.stack(P); mp = P.mean(0).argmax(1)
    return p, dict(density=gate.score(Z), confidence=-p.max(1),
                   entropy=-(p*np.log(p+1e-12)).sum(1),
                   margin=-(part[:,-1]-part[:,-2]),
                   energy=-(np.log(np.exp(lg/T).sum(1))*T),
                   ensemble=P.var(0)[np.arange(len(mp)), mp])

def aurc_legacy(sc, correct, n=60):
    o = np.argsort(sc); c = np.asarray(correct)[o]; cov = np.linspace(0.05, 1, n)
    risk = np.array([1-c[:max(1,int(round(f*len(c))))].mean() for f in cov])
    return float(np.trapezoid(risk, cov)/(cov[-1]-cov[0]))

def aurc(sc, correct, n=60):
    """Normalized partial AURC on [0.05,1], expected risk within exact score ties."""
    sc, correct = np.asarray(sc), np.asarray(correct, dtype=float)
    if len(sc) != len(correct) or not len(sc) or not np.isfinite(sc).all():
        raise ValueError("AURC requires aligned nonempty finite score arrays")
    order = np.argsort(sc, kind="stable"); scores, corr = sc[order], correct[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    means = np.add.reduceat(corr, starts) / (ends-starts)
    expected = np.repeat(means, ends-starts)
    cumulative = np.cumsum(expected)
    cov = np.linspace(0.05,1,n)
    counts = np.maximum(1, np.rint(cov*len(scores)).astype(int))
    risk = 1-cumulative[counts-1]/counts
    return float(np.trapezoid(risk,cov)/(cov[-1]-cov[0]))

def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); acc = (pred==y).astype(float)
    e, ed = 0.0, np.linspace(0,1,bins+1)
    for i in range(bins):
        m = (conf>ed[i]) & (conf<=ed[i+1])
        if m.sum(): e += m.mean()*abs(acc[m].mean()-conf[m].mean())
    return float(e)
#print("envelope and statistics defined")


# ===== CELL 12 =====
RNG = np.random.RandomState(0)
def f_rotation(X): return np.roll(X, 3, axis=2)
def f_dropout(X):
    Xc = X.copy()
    for i in range(len(Xc)): Xc[i][:, RNG.choice(8,3,replace=False)] = 0.0
    return Xc
def f_gain(X):    return X * RNG.uniform(0.3, 2.5, (len(X),1,8)).astype(np.float32)
def f_artifact(X):
    Xc = X.copy()
    for i in range(len(Xc)):
        pos = RNG.choice(X.shape[1], 6, replace=False)
        Xc[i][pos] += RNG.choice([-1,1],(6,8)).astype(np.float32)*8.0
    return Xc
def f_noise(X):   return X + RNG.randn(*X.shape).astype(np.float32)*2.0
FAULTS = [("broadband noise", f_noise), ("gain shift", f_gain),
          ("channel dropout", f_dropout), ("channel rotation", f_rotation),
          ("motion artefact", f_artifact)]
#print("faults:", [f[0] for f in FAULTS])


A=np.load(os.path.join(DATA, "d1.npz")); X1,y1,s1,ss1=A["X"],A["y"],A["subj"],A["sess"]
B=np.load(os.path.join(DATA, "d2.npz")); X2,y2,s2=B["X"],B["y"],B["subj"]
IDX = dict(
    source_train = np.where(np.isin(s1, range(1,21)) & np.isin(ss1, [1,2,3,4]))[0],
    source_test  = np.where(np.isin(s1, range(1,21)) & (ss1 == 5))[0],
    target       = np.where(np.isin(s1, range(21,31)))[0])
ST=IDX["source_train"]
MU, SD = norm_fit(X1[ST])
norm = lambda X: ((X - MU) / SD).astype(np.float32)
keep = np.isin(y1, list(D1_MAP)); ym = np.array([D1_MAP.get(v,-1) for v in y1])
sub3 = lambda name: IDX[name][keep[IDX[name]]]
