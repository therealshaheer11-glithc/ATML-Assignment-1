"""Recompute every derived number quoted in the report text for Tasks 3-4 and the
cross-task discussion, and assert it against the published result files.

Run from the repository root:  python report/scripts/check_report_numbers.py
Numbers that are read directly from a results table are not repeated here.
"""
import csv
import json

import numpy as np
from scipy.stats import spearmanr

T2 = "task2/results"
T3 = "task3/results"
T4 = "task4/results"
CLS = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
failures = 0


def check(name, got, want, tol=0.005):
    global failures
    ok = abs(got - want) <= tol
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}: {name}: {got:.6g} (report {want})")


def hist(path):
    return list(csv.DictReader(open(path)))


def clipped(rows):
    return sum(round(float(r["gradient_clipped_fraction"]) * 235) for r in rows), 235 * len(rows)


# ---------------- Task 3 ----------------
d = json.load(open(f"{T3}/final/final_results.json"))
R = d["results"]
runs = ["erm", "dan_dg_0p1", "dan_dg_1", "dan_dg_10", "sam"]
src = [R[m]["source_validation"]["mean_source_macro_f1"] for m in runs]
check("T3 Spearman(mean-source F1, Sketch F1), 5 runs",
      spearmanr(src, [R[m]["target"]["macro_f1"] for m in runs]).correlation, 1.00)
check("T3 Spearman(mean-source F1, Sketch acc), 5 runs",
      spearmanr(src, [R[m]["target"]["accuracy"] for m in runs]).correlation, 0.80)
cm = np.array(R["dan_dg_1"]["target"]["confusion_matrix"])
check("T3 DAN-DG dog errors", cm[0].sum() - cm[0, 0], 576, 0)
check("T3 DAN-DG dog->elephant", cm[0, 1], 337, 0)
check("T3 DAN-DG elephant predictions", cm[:, 1].sum(), 1431, 0)
check("T3 DAN-DG person correct", cm[6, 6], 14, 0)

P = list(csv.DictReader(open(f"{T3}/final/target_predictions.csv")))
for m, want in (("erm", 236), ("sam", 12)):
    n = sum(1 for r in P if r[f"{m}_correct"] == "False" and float(r[f"{m}_confidence"]) >= 0.9)
    check(f"T3 {m} Sketch errors with confidence >= 0.9", n, want, 0)

h1 = hist(f"{T3}/training/prescribed/dan_dg_1/history.csv")
c, n = clipped(h1)
check("T3 DAN-DG lambda=1 clipped updates", c, 1869, 0)
check("T3 DAN-DG lambda=1 total updates", n, 1880, 0)
c, n = clipped(hist(f"{T2}/training/source_only/history.csv"))
check("T3 ERM clipped share (%)", 100 * c / n, 1.8, 0.05)
c, n = clipped(hist(f"{T3}/training/supplementary/dan_dg_floor_1/history.csv"))
check("T3 floor lambda=1 clipped share (%)", 100 * c / n, 30.9, 0.05)
check("T3 DAN-DG lambda=1 grad norm epoch 1", float(h1[0]["gradient_norm"]), 157, 0.5)
check("T3 DAN-DG lambda=1 grad norm epoch 8", float(h1[7]["gradient_norm"]), 782, 0.5)
check("T3 P-A batch median epoch 1", float(h1[0]["mmd_median_photo__art_painting"]), 0.028, 0.0005)
check("T3 P-A batch median epoch 8", float(h1[7]["mmd_median_photo__art_painting"]), 0.00004, 0.000005)
check("T3 MMD epoch 1", float(h1[0]["mmd_loss"]), 0.368, 0.0005)
check("T3 MMD epoch 8", float(h1[7]["mmd_loss"]), 0.317, 0.0005)
late = [float(r["gradient_clipped_fraction"]) for r in hist(f"{T2}/training/dan_1/history.csv")[8:]]
check("T2 DAN late-epoch clipped share (about 90%)", 100 * np.mean(late), 90, 1.5)

# ---------------- Task 4 ----------------
f = json.load(open(f"{T4}/final_results.json"))
X = json.load(open(f"{T4}/extra_analysis.json"))
for s in ("msp", "mls", "energy", "mahalanobis"):
    check(f"T4 extra analysis reproduces {s} near AUROC",
          X["check"][f"vanilla:{s}:near_auroc"], f["table_1_vanilla_scores"][s]["near"]["auroc"], 1e-6)
for m in ("vanilla", "gcsc", "proser"):
    check(f"T4 extra analysis reproduces {m} CSA",
          X["check"][f"{m}:csa"], f["table_2_trained_models"][f"{m}:mls"]["closed_set_accuracy"], 0)
A = X["vanilla_score_agreement"]
check("T4 MLS/Energy near disagreements", A["near_mls_vs_energy"]["mls_only_accepts"] + A["near_mls_vs_energy"]["energy_only_accepts"], 27, 0)
check("T4 MLS/Energy far disagreements", A["far_mls_vs_energy"]["mls_only_accepts"] + A["far_mls_vs_energy"]["energy_only_accepts"], 34, 0)
check("T4 far accepted by MSP only (vs MLS)", A["far_msp_vs_mls"]["msp_only_accepts"], 112, 0)
check("T4 far accepted by MLS only (vs MSP)", A["far_msp_vs_mls"]["mls_only_accepts"], 20, 0)
check("T4 far accepted by MLS only (vs Mahalanobis)", A["far_mls_vs_mahalanobis"]["mls_only_accepts"], 70, 0)
check("T4 far accepted by Mahalanobis only", A["far_mls_vs_mahalanobis"]["mahalanobis_only_accepts"], 92, 0)
check("T4 Spearman MLS-Mahalanobis", A["spearman_all"]["mls-mahalanobis"], 0.60)
check("T4 Spearman MLS-Energy", A["spearman_all"]["mls-energy"], 1.00)
for g, want in (("known", 4.79), ("near", 27.12), ("far", 47.50)):
    check(f"T4 PROSER dummy wins on {g} (%)", 100 * X["proser_dummy"][g]["dummy_wins_fraction"], want, 0.01)
ph = hist(f"{T4}/training/proser/history.csv")
acc = [float(r["validation_accuracy"]) for r in ph]
check("T4 PROSER val acc epoch 1 (%)", 100 * acc[0], 94.96, 0.005)
check("T4 PROSER min val acc (%)", 100 * min(acc), 82.56, 0.005)
check("T4 PROSER min val acc epoch", acc.index(min(acc)) + 1, 29, 0)
check("T4 data-placeholder loss epoch 1", float(ph[0]["data_placeholder"]), 3.31, 0.005)
check("T4 data-placeholder loss epoch 50", float(ph[-1]["data_placeholder"]), 0.02, 0.005)

# ---------------- Task 2 values used in the cross-task discussion ----------------
t2 = {r["run_id"]: r for r in csv.DictReader(open(f"{T2}/final/all_six_comparison.csv"))}
check("T2 Source-only separability (%)", 100 * float(t2["source_only"]["domain_separability"]), 99.59)
check("T2 DAN separability (%)", 100 * float(t2["dan_1"]["domain_separability"]), 90.11)
check("T2 DAN lambda=10 separability (%)", 100 * float(t2["dan_10"]["domain_separability"]), 89.97)
check("T2 DAN lambda=10 Sketch acc (%)", 100 * float(t2["dan_10"]["target_accuracy"]), 4.07)
check("T2 DAN lambda=0.1 Sketch acc (%)", 100 * float(t2["dan_0p1"]["target_accuracy"]), 61.47)

print(f"\n{'ALL CHECKS PASS' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
raise SystemExit(failures != 0)
