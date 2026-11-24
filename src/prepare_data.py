"""Parse both public datasets into d1.npz (Dataset A) and d2.npz (Dataset B).
Usage: python prepare_data.py <dataset_A_root_with_subject_dirs> <dataset_B_repo_root>"""
import os, re, sys, glob, json, warnings, numpy as np, scipy.io as sio, pandas as pd
PARSE_ISSUES = []

OUT = os.environ.get("EMG_DATA", os.path.dirname(os.path.abspath(__file__)))
N_T, N_CH, N_CLASS = 1000, 8, 6

# ---- Dataset A: Hunter-qing Forearm sEMG Gesture Dataset (CC BY 4.0), stored rectified
def _parse_name(path):
    m = re.search(r"subject_(\d+)/session_(\d+)/(.+)\.mat$", path.replace("\\", "/"))
    subj, sess, name = int(m.group(1)), int(m.group(2)), m.group(3)
    g = re.match(r"gesture_(\d+)_(\d+)", name)
    if g: return subj, sess, int(g.group(1))
    r = re.match(r"(?:rest|relax)[_-]?(\d+)", name, re.I)
    return subj, sess, (0 if r else None)

def load_a(root):
    files = sorted(glob.glob(os.path.join(root, "subject_*", "session_*", "*.mat")))
    X, y, subj, sess = [], [], [], []
    for f in files:
        s, ss, lab = _parse_name(f)
        d = sio.loadmat(f); emg = np.asarray(d["emg"], dtype=np.float32)
        if emg.shape != (N_T, N_CH): continue
        L = np.unique(np.asarray(d["label"]).ravel())
        if L.size == 1: lab = int(L[0])
        if lab is None or not (0 <= lab < N_CLASS): continue
        X.append(emg); y.append(lab); subj.append(s); sess.append(ss)
    return np.stack(X), np.asarray(y, np.int64), np.asarray(subj, np.int64), np.asarray(sess, np.int64)

# ---- Dataset B: aljazfrancic myo-readings-dataset (MIT), signed bytes, continuous files
D2_MAP = {0: 0, 7: 1, 3: 2, 4: 2}     # rest, fist, radial/ulnar deviation -> wrist deviation
MAX_WIN_PER_LABEL = 2

def parse_b_file(path):
    try:
        a = pd.read_csv(path, header=None, dtype=np.int16).values
    except Exception as exc:
        PARSE_ISSUES.append({"file": str(path), "reason": str(exc)})
        warnings.warn(f"Excluded unreadable file {path}: {exc}")
        return []
    if a.shape[1] != 9 or len(a) < N_T:
        PARSE_ISSUES.append({"file": str(path), "reason": f"invalid shape {a.shape}"})
        return []
    emg, lab = a[:, :8].astype(np.float32), a[:, 8]          # cast before any absolute value
    out, i, taken, n = [], 0, {}, len(lab)
    while i <= n - N_T:  # include a full window ending at the final sample
        L = lab[i]
        if L not in D2_MAP: i += N_T // 2; continue
        if (lab[i:i + N_T] == L).all():                     # window must not cross a label change
            m = D2_MAP[L]
            if taken.get(m, 0) < MAX_WIN_PER_LABEL:
                out.append((emg[i:i + N_T], m)); taken[m] = taken.get(m, 0) + 1
            i += N_T
        else:
            i += N_T // 4
    return out

def load_b(root):
    X, y, subj, sess = [], [], [], []
    for sd in sorted(glob.glob(os.path.join(root, "_readings_right_hand", "*-*"))):
        bn = os.path.basename(sd)
        try: pid, sn = int(bn.split("-")[0]), int(bn.split("-")[1])
        except Exception: continue
        for f in sorted(glob.glob(os.path.join(sd, "*.txt"))):
            for w, m in parse_b_file(f):
                X.append(w); y.append(m); subj.append(pid); sess.append(sn)
    return np.stack(X), np.asarray(y, np.int64), np.asarray(subj, np.int64), np.asarray(sess, np.int64)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, y, s, ss = load_a(sys.argv[1])
    np.savez_compressed(os.path.join(OUT, "d1.npz"), X=X, y=y, subj=s, sess=ss)
    print(f"Dataset A: {X.shape}, range {X.min():.0f} to {X.max():.0f}, class counts {np.bincount(y).tolist()}")
    X, y, s, ss = load_b(sys.argv[2])
    np.savez_compressed(os.path.join(OUT, "d2.npz"), X=X, y=y, subj=s, sess=ss)
    with open(os.path.join(OUT, "parse_issues.json"), "w") as f: json.dump(PARSE_ISSUES, f, indent=2)
    print(f"Dataset B: {X.shape}, range {X.min():.0f} to {X.max():.0f}, participants {len(np.unique(s))}, "
          f"class counts {np.bincount(y).tolist()}")
