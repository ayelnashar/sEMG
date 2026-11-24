"""Revised paper, Dataset A main split, ten seeds, calibrated linear probe."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
import time, pickle
from core import *
import core
from lib2 import make_conditions, contract_score, td_features, LDABaseline
from lib3 import *

OUT = os.environ.get("OUT", os.path.join(RES, "v3_a")); os.makedirs(OUT, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46,47,48,49,50,51").split(",")]
is_cal = np.isin(s1[ST], CALIB_SUBJECTS); FIT, CAL = ST[~is_cal], ST[is_cal]
TE, TG = IDX["source_test"], IDX["target"]
LO, HI = float(X1[ST].min()), float(X1[ST].max())
CONDS = make_conditions(lambda X: X, MU, SD, False, True); CONDS.pop("matched")
FAULT_FN = dict(core.FAULTS)
POOL = list(FAULT_FN) + ["gain x4", "gain x0.25", "envelope 150 ms", "per-window z-score", "400 Hz stream"]
Xn1 = norm(X1)

def apply_pool(Xraw, names):
    X = Xraw.copy(); post = False
    for nm in names:
        if nm in FAULT_FN: X = FAULT_FN[nm](X)
        elif nm == "per-window z-score": post = True
        else: X = CONDS[nm](X) * SD + MU
    return CONDS["per-window z-score"](X) if post else norm(X)

def shifted(nm, f):
    Xin = norm(f(X1[TE])) if nm in FAULT_FN else f(X1[TE])
    return Xin

def run(seed):
    fn = f"{OUT}/a_{seed}.pkl"
    if os.path.exists(fn): return
    t0 = time.time(); core.RNG = np.random.RandomState(seed)
    enc, rec = pretrain(Xn1[ST], seed=seed)
    torch.save(enc.state_dict(), f"{OUT}/enc_{seed}.pt")
    Zr = embed(enc, Xn1); m0, s0 = Zr[ST].mean(0), Zr[ST].std(0) + 1e-6
    Zs = lambda Xin: ((embed(enc, Xin) - m0) / s0).astype(np.float32)
    Z = ((Zr - m0) / s0).astype(np.float32)
    base = fit_probe(Z[FIT], y1[FIT], Z[CAL], y1[CAL], K6, seed)
    gate = Gate().fit(Z[FIT], y1[FIT], K6)
    R = dict(seed=seed, recon=rec, T=float(base.logT.exp()))

    # six-class ladder
    for nm, v in [("source_test", TE), ("target", TG)]:
        p, sc = pstats(base, gate, Z[v]); R[nm] = summary(p, y1[v], K6, gate, Z[v])
        if nm == "target":
            corr = (p.argmax(1) == y1[v]).astype(float)
            R[nm]["aurc"] = {k: aurc(sc[k], corr) for k in STATS}
            R[nm]["scores"] = {k: sc[k].astype(np.float32) for k in STATS}
            R[nm]["correct"] = corr.astype(np.float32)
            R["per_participant"] = {int(j): dict(acc=float(corr[s1[v] == j].mean()),
                                                 bal=float(balanced_accuracy_score(y1[v][s1[v] == j], p.argmax(1)[s1[v] == j])),
                                                 conf=float(p.max(1)[s1[v] == j].mean())) for j in range(21, 31)}
        else:
            sc_in = sc; sc_in["range check"] = contract_score(Xn1[TE], MU, SD, LO, HI)

    # untrained encoder control
    seed_all(seed); encr = EMGEncoder().to(DEV).eval()
    Zu = embed(encr, Xn1); Zu = ((Zu - Zu[ST].mean(0)) / (Zu[ST].std(0) + 1e-6)).astype(np.float32)
    bu = fit_probe(Zu[FIT], y1[FIT], Zu[CAL], y1[CAL], K6, seed); gu = Gate().fit(Zu[FIT], y1[FIT], K6)
    pu, su = pstats(bu, gu, Zu[TG]); cu = (pu.argmax(1) == y1[TG]).astype(float)
    R["untrained"] = dict(acc=float(cu.mean()), bal=float(balanced_accuracy_score(y1[TG], pu.argmax(1))),
                          aurc={k: aurc(su[k], cu) for k in STATS})

    # faults and mismatch family, six-class
    R["shift"] = {}
    for nm, f in list(FAULT_FN.items()) + list(CONDS.items()):
        Xin = shifted(nm, f); Zo = Zs(Xin)
        po, so = pstats(base, gate, Zo); so["range check"] = contract_score(Xin, MU, SD, LO, HI)
        R["shift"][nm] = dict(**summary(po, y1[TE], K6, gate, Zo),
                              auroc={k: auroc(sc_in[k], so[k]) for k in so},
                              tpr5={k: tpr_at(sc_in[k], so[k]) for k in STATS})

    # three-class ladder with Dataset B and the factorial
    st3 = sub3("source_train"); ic = np.isin(s1[st3], CALIB_SUBJECTS)
    b3 = fit_probe(Z[st3[~ic]], ym[st3[~ic]], Z[st3[ic]], ym[st3[ic]], KC, seed)
    g3 = Gate().fit(Z[st3[~ic]], ym[st3[~ic]], KC)
    muBr, sdBr = norm_fit(np.abs(X2)); muBs, sdBs = norm_fit(X2)
    L = {}; ref = None
    for nm, Xin, yq in [("source_test", Xn1[sub3("source_test")], ym[sub3("source_test")]),
                        ("cross-participant", Xn1[sub3("target")], ym[sub3("target")]),
                        ("cross-dataset", norm(np.abs(X2)), y2), ("mismatch", norm(X2), y2),
                        ("signed + own norm", ((X2 - muBs) / sdBs).astype(np.float32), y2),
                        ("rectified + own norm", ((np.abs(X2) - muBr) / sdBr).astype(np.float32), y2)]:
        Zq = Zs(Xin); p3, s3 = pstats(b3, g3, Zq); s3["range check"] = contract_score(Xin, MU, SD, LO, HI)
        r = summary(p3, yq, KC, g3, Zq)
        if ref is None: ref = s3
        else:
            r["auroc"] = {k: auroc(ref[k], s3[k]) for k in s3}
            r["tpr5"] = {k: tpr_at(ref[k], s3[k]) for k in STATS}
        L[nm] = r
    R["ladder3"] = L

    # cascade on cross-participant windows
    rng = np.random.RandomState(seed + 7); n = len(TG)
    faulty = rng.rand(n) < 0.5
    combos = [list(rng.choice(POOL, rng.randint(1, 3), replace=False)) if faulty[i] else [] for i in range(n)]
    Xin = np.stack([apply_pool(X1[TG][i:i+1], combos[i])[0] for i in range(n)])
    Zm = Zs(Xin); pm, sm = pstats(base, gate, Zm)
    chk = contract_score(Xin, MU, SD, LO, HI) > 0
    pc, _ = pstats(base, gate, Z[CAL]); cthr = np.percentile(pc.max(1), 10)
    conf_ok = pm.max(1) >= cthr; dens_ok = ~gate.refuse(Zm)
    pol = {"none": np.ones(n, bool), "confidence": conf_ok, "density": dens_ok, "range check": ~chk,
           "range + density": ~chk & dens_ok, "range + density + confidence": ~chk & dens_ok & conf_ok}
    yt = y1[TG]; pred = pm.argmax(1); C = {}
    for k, a in pol.items():
        C[k] = dict(cov_clean=float(a[~faulty].mean()), cov_faulty=float(a[faulty].mean()),
                    bal_accepted=float(balanced_accuracy_score(yt[a], pred[a])),
                    err_accepted=float((pred[a] != yt[a]).mean()),
                    harmful=float((a & (pred != yt) & faulty).mean()),
                    harmful_share=float((a & (pred != yt) & faulty).sum() / max(1, a.sum())))
    R["cascade"] = C; R["cascade_meta"] = dict(n=n, n_faulty=int(faulty.sum()), conf_thr=float(cthr),
                                               err_faulty_none=float((pred[faulty] != yt[faulty]).mean()),
                                               err_clean_none=float((pred[~faulty] != yt[~faulty]).mean()))
    if seed == SEEDS[0]:
        x = torch.from_numpy(Xn1[:1]).to(DEV); enc.eval()
        with torch.no_grad():
            for _ in range(5): enc(x)
            t1 = time.perf_counter()
            for _ in range(300): enc(x)
            lat = (time.perf_counter() - t1) / 300
        zz = Z[:1]; t1 = time.perf_counter()
        for _ in range(3000): gate.score(zz)
        R["latency"] = dict(encoder_ms=lat * 1e3, gate_ms=(time.perf_counter() - t1) / 3 ,
                            params=int(sum(q.numel() for q in enc.parameters())))
    pickle.dump(R, open(fn, "wb"))
    mm = R["ladder3"]["mismatch"]
    print(f"[A {seed}] {time.time()-t0:.0f}s tgt acc {R['target']['acc']:.3f} | mismatch bal {mm['bal']:.3f} conf {mm['conf']:.4f} "
          f"AUROC conf {mm['auroc']['confidence']:.3f} dens {mm['auroc']['density']:.3f} | cascade harmful "
          f"{C['none']['harmful']:.3f}->{C['range + density + confidence']['harmful']:.3f}", flush=True)

for s in SEEDS: run(s)
print("A DONE", flush=True)
