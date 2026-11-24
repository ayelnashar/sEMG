import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.environ.get("EMG_RES", os.path.join(HERE, "results"))
import pickle
from core import *; from lib3 import fit_probe
res = {}
for seed in [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]:
    enc = EMGEncoder().to(DEV); enc.load_state_dict(torch.load(os.path.join(RES, "v3_a", f"enc_{seed}.pt"), map_location=DEV)); enc.eval()
    Xn1 = norm(X1); Zr = embed(enc, Xn1); Z = ((Zr - Zr[ST].mean(0)) / (Zr[ST].std(0) + 1e-6)).astype(np.float32)
    wa = []
    for sj in range(1, 21):
        tr = ST[s1[ST] == sj]; te = IDX["source_test"][s1[IDX["source_test"]] == sj]
        perm = np.random.RandomState(sj).permutation(len(tr)); nc = max(4, int(0.25 * len(tr)))
        b = fit_probe(Z[tr[perm[nc:]]], y1[tr[perm[nc:]]], Z[tr[perm[:nc]]], y1[tr[perm[:nc]]], K6, seed)
        with torch.no_grad(): pr = b(torch.from_numpy(Z[te])).numpy().argmax(1)
        wa.append(float((pr == y1[te]).mean()))
    res[seed] = wa; print(seed, np.mean(wa), np.std(wa))
pickle.dump(res, open(os.path.join(RES, "v3_within.pkl"), "wb"))
