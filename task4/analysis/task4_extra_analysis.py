"""Task 4 post-evaluation analysis from the saved (already locked) outputs.

Reads only the cached logits/features written by evaluate_osr.py, the frozen
evaluation lock, and the frozen Mahalanobis statistics. It changes nothing and
re-runs no model. Prints one JSON block to paste back.

Colab usage (paths are the ones used in the Task 4 README):
    !python task4_extra_analysis.py \
        --outputs /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/final_evaluation/outputs \
        --lock /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/evaluation_lock/evaluation_lock.json \
        --stats /content/drive/MyDrive/ATML-PA1/task4_open_set_20260925/evaluation_lock/mahalanobis_stats.npz
"""
import argparse, json
from pathlib import Path
import numpy as np

CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# CIFAR-100 fine-label indices (torchvision alphabetical order).
C100 = {9: "bottle", 10: "bowl", 13: "bus", 15: "camel", 20: "chair", 22: "clock", 34: "fox",
        39: "keyboard", 42: "leopard", 48: "motorcycle", 51: "mushroom", 58: "pickup_truck",
        82: "sunflower", 89: "tractor", 94: "wardrobe", 97: "wolf"}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logsumexp(z):
    m = z.max(1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]


def scores(d, stats):
    z = d["known_logits"].astype(np.float64)
    out = {"msp": 1 - softmax(z).max(1), "mls": -z.max(1), "energy": -logsumexp(z)}
    if stats is not None:
        f = d["features"].astype(np.float64)
        mu, var = stats["class_means"].astype(np.float64), stats["shared_variance"].astype(np.float64)
        dist = np.stack([((f - m) ** 2 / var).sum(1) for m in mu], 1)
        out["mahalanobis"] = dist.min(1)
    return out


def auroc(known, unknown):
    # P(unknown score > known score) + 0.5 ties; larger = more unknown.
    s = np.concatenate([known, unknown])
    r = s.argsort().argsort().astype(np.float64) + 1
    # average ranks for ties
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    ranks = np.empty_like(r); i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n2 = len(known), len(unknown)
    return float((ranks[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2))


def spearman(a, b):
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    lock = json.load(open(a.lock))
    stats = np.load(a.stats)
    th = lock["thresholds"]
    O = {m: {g: np.load(Path(a.outputs) / f"{m}_{g}.npz") for g in ("known", "near", "far")}
         for m in ("vanilla", "gcsc", "proser")}
    S = {m: {g: scores(O[m][g], stats if m == "vanilla" else None) for g in O[m]} for m in O}
    res = {"check": {}}

    # sanity: reproduce locked table values
    for sc in ("msp", "mls", "energy", "mahalanobis"):
        res["check"][f"vanilla:{sc}:near_auroc"] = round(auroc(S["vanilla"]["known"][sc], S["vanilla"]["near"][sc]), 6)
    for m in ("vanilla", "gcsc", "proser"):
        k = O[m]["known"]
        res["check"][f"{m}:csa"] = float((k["known_logits"].argmax(1) == k["labels"]).mean())

    # 1) per-unknown-class acceptance and absorbing CIFAR-10 labels
    per = {}
    for g in ("near", "far"):
        lab = O["vanilla"][g]["labels"]
        for cid in sorted(set(lab.tolist())):
            name = C100.get(cid, str(cid)); idx = lab == cid
            row = {"group": g, "n": int(idx.sum())}
            for m in ("vanilla", "gcsc", "proser"):
                acc = S[m][g]["mls"][idx] <= th[f"{m}:mls"]
                row[f"{m}_mls_accepted"] = int(acc.sum())
                row[f"{m}_mls_auroc"] = round(auroc(S[m]["known"]["mls"], S[m][g]["mls"][idx]), 4)
            for sc in ("msp", "energy", "mahalanobis"):
                row[f"vanilla_{sc}_accepted"] = int((S["vanilla"][g][sc][idx] <= th[f"vanilla:{sc}"]).sum())
            accv = S["vanilla"][g]["mls"][idx] <= th["vanilla:mls"]
            pred = O["vanilla"][g]["known_logits"][idx].argmax(1)[accv]
            cnt = np.bincount(pred, minlength=10)
            top = np.argsort(-cnt)[:3]
            row["vanilla_mls_absorbed_by"] = {CIFAR10[t]: int(cnt[t]) for t in top if cnt[t] > 0}
            predall = O["vanilla"][g]["known_logits"][idx].argmax(1)
            cnta = np.bincount(predall, minlength=10)
            row["vanilla_all_predicted_as"] = {CIFAR10[t]: int(cnta[t]) for t in np.argsort(-cnta)[:2]}
            per[name] = row
    res["per_unknown_class"] = per

    # overall absorbing labels among accepted unknowns (vanilla MLS)
    for g in ("near", "far"):
        acc = S["vanilla"][g]["mls"] <= th["vanilla:mls"]
        cnt = np.bincount(O["vanilla"][g]["known_logits"].argmax(1)[acc], minlength=10)
        res[f"{g}_accepted_absorbed_by_vanilla_mls"] = {CIFAR10[t]: int(cnt[t]) for t in np.argsort(-cnt) if cnt[t] > 0}

    # 2) score agreement on vanilla
    agree = {}
    names = ("msp", "mls", "energy", "mahalanobis")
    for scope, groups in (("all", ("known", "near", "far")), ("unknown_only", ("near", "far"))):
        cat = {s: np.concatenate([S["vanilla"][g][s] for g in groups]) for s in names}
        agree[f"spearman_{scope}"] = {f"{x}-{y}": round(spearman(cat[x], cat[y]), 3)
                                      for i, x in enumerate(names) for y in names[i + 1:]}
    for g in ("near", "far"):
        acc = {s: S["vanilla"][g][s] <= th[f"vanilla:{s}"] for s in names}
        for x, y in (("msp", "mls"), ("mls", "mahalanobis"), ("mls", "energy"), ("msp", "mahalanobis")):
            agree[f"{g}_{x}_vs_{y}"] = {"both_accept": int((acc[x] & acc[y]).sum()),
                                        f"{x}_only_accepts": int((acc[x] & ~acc[y]).sum()),
                                        f"{y}_only_accepts": int((~acc[x] & acc[y]).sum()),
                                        "both_reject": int((~acc[x] & ~acc[y]).sum())}
        # among unknowns MLS accepts but Mahalanobis rejects: mean max softmax
    k = S["vanilla"]["known"]
    agree["known_accept_counts"] = {s: int((k[s] <= th[f"vanilla:{s}"]).sum()) for s in names}
    res["vanilla_score_agreement"] = agree

    # 3) logit magnitude / confidence summaries per model and group
    mag = {}
    for m in O:
        for g in O[m]:
            z = O[m][g]["known_logits"].astype(np.float64)
            mag[f"{m}_{g}"] = {"mean_max_logit": round(float(z.max(1).mean()), 3),
                               "mean_msp": round(float(softmax(z).max(1).mean()), 4)}
    res["confidence"] = mag

    # 4) per-class closed-set accuracy
    csa = {}
    for m in O:
        k = O[m]["known"]; p = k["known_logits"].argmax(1); y = k["labels"]
        csa[m] = {CIFAR10[c]: round(float((p[y == c] == c).mean()), 4) for c in range(10)}
    res["per_class_csa"] = csa

    # 5) PROSER dummy responses
    b = lock["proser_dummy_bias"]; pr = {}
    for g in ("known", "near", "far"):
        d = O["proser"][g]
        wins = (d["dummy_logits"].max(1) + b) > d["known_logits"].max(1)
        pr[g] = {"dummy_wins_fraction": round(float(wins.mean()), 4),
                 "mean_max_dummy_plus_bias": round(float((d["dummy_logits"].max(1) + b).mean()), 3),
                 "mean_max_known": round(float(d["known_logits"].max(1).mean()), 3)}
    res["proser_dummy"] = pr

    # 6) Vanilla vs PROSER/GCSC: accepted near unknowns, per model, overall
    res["accepted_counts_mls"] = {m: {g: int((S[m][g]["mls"] <= th[f"{m}:mls"]).sum()) for g in ("near", "far")} for m in O}
    print("=== PASTE EVERYTHING BELOW ===")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
