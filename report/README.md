# Report evidence

Scripts and figures used by the PDF report that are not produced by the task pipelines
themselves. Everything here reads only published result files; nothing retrains a model
or re-evaluates a target or unknown set.

Run from the repository root:

```bash
python report/scripts/check_report_numbers.py   # recomputes every derived number quoted for Tasks 3-4 and the cross-task discussion
python report/scripts/make_task3_figures.py     # Task 3 figures -> report/figures/task3/
python report/scripts/make_task4_figure.py      # Task 4 appendix training-curve figure -> report/figures/task4/
```

The Task 4 score-distribution figure is the one written by the locked evaluation,
`task4/results/vanilla_score_distributions.png`.

## Task 4 per-class analysis

`task4/analysis/task4_extra_analysis.py` reads the cached logits and features saved by
the one-time Task 4 evaluation (kept outside Git), the frozen evaluation lock, and the
frozen Mahalanobis statistics. It changes nothing. Its output is published as
`task4/results/extra_analysis.json` and supplies the per-class acceptance, absorbing
labels, score agreement, and PROSER dummy-response numbers in the report. The file first
reproduces the locked near AUROCs and closed-set accuracies as a consistency check, which
`check_report_numbers.py` verifies.
