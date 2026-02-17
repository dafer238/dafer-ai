Capstone — Project Guide

Goal
This capstone ties together forecasting, uncertainty quantification, and deployment. Produce a short technical report, reproducible code, and a demo showing an end-to-end system.

Expected Deliverables
- Problem statement: clear goals, evaluation metrics, and dataset description.
- Baseline model: simple model with evaluation and diagnostics.
- Deep model: more advanced model (e.g., deep NN or transformer) with training + ablation.
- Uncertainty analysis: calibration, intervals, or ensembling; discussion of failure modes.
- Notebook(s): reproducible notebooks for experiments and figures.
- Deployment demo: lightweight API or runnable script that serves predictions.
- Report: `report.md` summarizing approach, results, and lessons learned (2–4 pages).

Suggested Workflow (milestones)
1) Week 1–4: Define problem, gather data, build baseline, and run initial experiments.
2) Week 5–8: Implement probabilistic / uncertainty approaches and document diagnostics.
3) Week 9–12: Build deep model, run ablations, and optimize training.
4) Week 13–16: Finalize uncertainty analysis, build deployment demo, and write report.

Evaluation criteria
- Reproducibility: can another user run notebooks and reproduce main results?
- Rigor: clear evaluation, diagnostics, and ablation studies.
- Clarity: concise report explaining design choices and failure modes.
- Impact: demonstration that the model or pipeline provides value for the chosen problem.

Ideas / Project templates
- Energy optimization: forecast demand, quantify uncertainty, and propose control actions.
- Trading signal: short-term return prediction with uncertainty-aware risk control.
- Industrial fault detection: sensor-based anomaly detection with calibrated predictions.

Notes
- Keep experiments small and reproducible; prefer clear diagnostics to marginal accuracy gains.
- Start with simple baselines and iterate; incremental progress is expected.
