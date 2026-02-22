"""
Add a 'Theory & References' markdown cell after the title cell of each notebook.
The cell links to the week's own theory.md and to prerequisite/related theory pages.
"""

import json
from pathlib import Path

BASE = Path(__file__).parent.resolve()

# For each notebook: (path, theory_refs)
# theory_refs is a list of (display_text, relative_link) tuples
# The first entry is always the current week's own theory
NOTEBOOKS = [
    (
        "02_fundamentals/week01_optimization/starter.ipynb",
        [
            ("📖 **Week 01 Theory** — Optimization", "theory.md"),
            (
                "Week 00a — AI Landscape (training loop overview)",
                "../../01_intro/week00_ai_landscape/theory.md#7-the-training-loop",
            ),
            (
                "Week 00b — Math & Data (calculus prerequisites)",
                "../../01_intro/week00b_math_and_data/theory.md#3-part-ii-calculus-and-optimisation",
            ),
        ],
    ),
    (
        "02_fundamentals/week02_advanced_optimizers/starter.ipynb",
        [
            ("📖 **Week 02 Theory** — Advanced Optimizers", "theory.md"),
            (
                "Week 01 — Gradient descent, SGD, momentum",
                "../week01_optimization/theory.md#4-gradient-descent",
            ),
            (
                "Week 00b — Exponential moving averages",
                "../../01_intro/week00b_math_and_data/theory.md#3-part-ii-calculus-and-optimisation",
            ),
        ],
    ),
    (
        "02_fundamentals/week03_linear_models/starter.ipynb",
        [
            ("📖 **Week 03 Theory** — Linear Models", "theory.md"),
            (
                "Week 01 — Gradient descent for regression",
                "../week01_optimization/theory.md#4-gradient-descent",
            ),
            (
                "Week 00b — Linear algebra & calculus",
                "../../01_intro/week00b_math_and_data/theory.md#2-part-i-linear-algebra",
            ),
        ],
    ),
    (
        "02_fundamentals/week04_dimensionality_reduction/starter.ipynb",
        [
            ("📖 **Week 04 Theory** — Dimensionality Reduction & PCA", "theory.md"),
            (
                "Week 03 — Covariance, linear models",
                "../week03_linear_models/theory.md#3-linear-regression",
            ),
            (
                "Week 00b — Eigendecomposition & SVD",
                "../../01_intro/week00b_math_and_data/theory.md#27-eigendecomposition",
            ),
        ],
    ),
    (
        "02_fundamentals/week05_clustering/starter.ipynb",
        [
            ("📖 **Week 05 Theory** — Clustering", "theory.md"),
            (
                "Week 04 — PCA for cluster visualisation",
                "../week04_dimensionality_reduction/theory.md#4-principal-component-analysis-the-optimisation-view",
            ),
            (
                "Week 00b — Distance metrics & norms",
                "../../01_intro/week00b_math_and_data/theory.md#26-norms",
            ),
        ],
    ),
    (
        "02_fundamentals/week06_regularization/starter.ipynb",
        [
            ("📖 **Week 06 Theory** — Regularization & Validation", "theory.md"),
            (
                "Week 03 — Linear regression (Ridge/Lasso modify this)",
                "../week03_linear_models/theory.md#3-linear-regression",
            ),
            (
                "Week 04 — SVD for Ridge interpretation",
                "../week04_dimensionality_reduction/theory.md#5-pca-via-the-singular-value-decomposition",
            ),
        ],
    ),
    (
        "03_probability/week07_likelihood/starter.ipynb",
        [
            ("📖 **Week 07 Theory** — Likelihood & MLE", "theory.md"),
            (
                "Week 03 — Linear regression & MSE",
                "../../02_fundamentals/week03_linear_models/theory.md#3-linear-regression",
            ),
            (
                "Week 06 — Regularization as MAP",
                "../../02_fundamentals/week06_regularization/theory.md#3-ridge-regression-l2-regularisation",
            ),
            (
                "Week 00b — Probability distributions",
                "../../01_intro/week00b_math_and_data/theory.md#4-part-iii-probability-and-statistics",
            ),
        ],
    ),
    (
        "03_probability/week08_uncertainty/starter.ipynb",
        [
            ("📖 **Week 08 Theory** — Uncertainty & Bayesian Inference", "theory.md"),
            (
                "Week 07 — Likelihood, MLE, Bayes' theorem",
                "../week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle",
            ),
            (
                "Week 06 — Ridge as Bayesian MAP",
                "../../02_fundamentals/week06_regularization/theory.md#35-bayesian-interpretation",
            ),
            (
                "Week 03 — Linear regression",
                "../../02_fundamentals/week03_linear_models/theory.md#3-linear-regression",
            ),
        ],
    ),
    (
        "03_probability/week09_time_series/starter.ipynb",
        [
            ("📖 **Week 09 Theory** — Time Series", "theory.md"),
            (
                "Week 03 — Linear regression (AR models extend this)",
                "../../02_fundamentals/week03_linear_models/theory.md#3-linear-regression",
            ),
            (
                "Week 06 — Time-series cross-validation",
                "../../02_fundamentals/week06_regularization/theory.md#7-time-series-cross-validation",
            ),
            (
                "Week 07–08 — Likelihood & residual diagnostics",
                "../week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle",
            ),
        ],
    ),
    (
        "03_probability/week10_surrogate_models/starter.ipynb",
        [
            ("📖 **Week 10 Theory** — Gaussian Processes & Bayesian Optimisation", "theory.md"),
            (
                "Week 08 — Bayesian linear regression (GP generalises this)",
                "../week08_uncertainty/theory.md#8-bayesian-linear-regression",
            ),
            (
                "Week 07 — Marginal likelihood",
                "../week07_likelihood/theory.md#4-maximum-likelihood-estimation-mle",
            ),
            (
                "Week 03 — Linear regression & polynomial features",
                "../../02_fundamentals/week03_linear_models/theory.md#5-polynomial-regression-and-feature-expansion",
            ),
        ],
    ),
    (
        "04_neural_networks/week11_nn_from_scratch/starter.ipynb",
        [
            ("📖 **Week 11 Theory** — Neural Networks From Scratch", "theory.md"),
            (
                "Week 00b — Chain rule",
                "../../01_intro/week00b_math_and_data/theory.md#36-the-chain-rule-in-depth",
            ),
            (
                "Week 01–02 — Gradient descent & Adam",
                "../../02_fundamentals/week01_optimization/theory.md#4-gradient-descent",
            ),
            (
                "Week 03 — Linear regression as a single neuron",
                "../../02_fundamentals/week03_linear_models/theory.md#3-linear-regression",
            ),
        ],
    ),
    (
        "04_neural_networks/week12_training_pathologies/starter.ipynb",
        [
            ("📖 **Week 12 Theory** — Training Pathologies", "theory.md"),
            (
                "Week 11 — Backpropagation & initialization",
                "../week11_nn_from_scratch/theory.md#5-backpropagation",
            ),
        ],
    ),
    (
        "05_deep_learning/week13_pytorch_basics/starter.ipynb",
        [
            ("📖 **Week 13 Theory** — PyTorch Fundamentals", "theory.md"),
            (
                "Week 11 — Backpropagation (autograd computes this)",
                "../../04_neural_networks/week11_nn_from_scratch/theory.md#5-backpropagation",
            ),
            (
                "Week 12 — BatchNorm, initialization, gradient clipping",
                "../../04_neural_networks/week12_training_pathologies/theory.md#7-fix-2-batch-normalisation",
            ),
        ],
    ),
    (
        "05_deep_learning/week14_training_at_scale/starter.ipynb",
        [
            ("📖 **Week 14 Theory** — Training at Scale", "theory.md"),
            (
                "Week 13 — PyTorch Dataset, DataLoader, training loop",
                "../week13_pytorch_basics/theory.md#7-data-loading-dataset-and-dataloader",
            ),
            (
                "Week 01–02 — Learning rate & SGD variance",
                "../../02_fundamentals/week01_optimization/theory.md#8-learning-rate-selection",
            ),
        ],
    ),
    (
        "05_deep_learning/week15_cnn_representations/starter.ipynb",
        [
            ("📖 **Week 15 Theory** — CNNs & Representation Learning", "theory.md"),
            (
                "Week 11 — Backpropagation (convolution is backpropagated the same way)",
                "../../04_neural_networks/week11_nn_from_scratch/theory.md#5-backpropagation",
            ),
            (
                "Week 12 — BatchNorm, residual connections",
                "../../04_neural_networks/week12_training_pathologies/theory.md#10-fix-5-residual-skip-connections",
            ),
            (
                "Week 13 — PyTorch nn.Module, training loop",
                "../week13_pytorch_basics/theory.md#4-building-models-with-nnmodule",
            ),
        ],
    ),
    (
        "05_deep_learning/week16_regularization_dl/starter.ipynb",
        [
            ("📖 **Week 16 Theory** — Regularization for Deep Learning", "theory.md"),
            (
                "Week 06 — L1/L2 regularization, cross-validation",
                "../../02_fundamentals/week06_regularization/theory.md#3-ridge-regression-l2-regularisation",
            ),
            (
                "Week 12 — BatchNorm (regularising side effect)",
                "../../04_neural_networks/week12_training_pathologies/theory.md#7-fix-2-batch-normalisation",
            ),
            ("Week 15 — CNN to regularise", "../week15_cnn_representations/theory.md"),
        ],
    ),
    (
        "06_sequence_models/week17_attention/starter.ipynb",
        [
            ("📖 **Week 17 Theory** — Attention Mechanisms", "theory.md"),
            (
                "Week 13 — PyTorch nn.Module, training loop",
                "../../05_deep_learning/week13_pytorch_basics/theory.md#4-building-models-with-nnmodule",
            ),
            (
                "Week 12 — Vanishing gradients (motivation for attention)",
                "../../04_neural_networks/week12_training_pathologies/theory.md#3-vanishing-gradients",
            ),
        ],
    ),
    (
        "06_sequence_models/week18_transformers/starter.ipynb",
        [
            ("📖 **Week 18 Theory** — Transformers", "theory.md"),
            (
                "Week 17 — Scaled dot-product & multi-head attention",
                "../week17_attention/theory.md#4-scaled-dot-product-attention",
            ),
            (
                "Week 12 — Residual connections & LayerNorm",
                "../../04_neural_networks/week12_training_pathologies/theory.md#10-fix-5-residual-skip-connections",
            ),
        ],
    ),
    (
        "07_transfer_learning/week19_finetuning/starter.ipynb",
        [
            ("📖 **Week 19 Theory** — Fine-Tuning & Transfer Learning", "theory.md"),
            (
                "Week 18 — Transformer architecture",
                "../../06_sequence_models/week18_transformers/theory.md#2-the-transformer-at-a-glance",
            ),
            (
                "Week 15 — CNNs (vision transfer learning)",
                "../../05_deep_learning/week15_cnn_representations/theory.md#10-transfer-learning-and-pretrained-features",
            ),
            (
                "Week 06/16 — Regularization techniques",
                "../../02_fundamentals/week06_regularization/theory.md",
            ),
        ],
    ),
    (
        "08_deployment/week20_deployment/starter.ipynb",
        [
            ("📖 **Week 20 Theory** — Deployment", "theory.md"),
            (
                "Week 14 — Checkpointing, mixed precision",
                "../../05_deep_learning/week14_training_at_scale/theory.md#5-robust-checkpointing",
            ),
            (
                "Week 19 — Fine-tuned model to deploy",
                "../../07_transfer_learning/week19_finetuning/theory.md",
            ),
        ],
    ),
]


def add_crossref_cells():
    count = 0
    for rel_path, refs in NOTEBOOKS:
        nb_path = BASE / rel_path
        if not nb_path.exists():
            print(f"  [SKIP] {rel_path} not found")
            continue

        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])

        # Check if a cross-ref cell already exists (avoid duplicates)
        for cell in cells[:5]:
            if cell.get("cell_type") == "markdown":
                src = "".join(cell.get("source", []))
                if "Theory & References" in src or "📖" in src:
                    print(f"  [SKIP] {rel_path}: cross-ref cell already exists")
                    break
        else:
            # Build the markdown content
            lines = ["---\n", "### 📚 Theory & References\n", "\n"]
            for display, link in refs:
                lines.append(f"- [{display}]({link})\n")
            lines.append("\n---")

            # Create the new cell
            new_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": lines,
            }

            # Insert after the first markdown cell (title + description)
            insert_idx = 1
            for i, cell in enumerate(cells):
                if cell.get("cell_type") == "markdown":
                    insert_idx = i + 1
                    break

            cells.insert(insert_idx, new_cell)
            nb["cells"] = cells

            nb_path.write_text(
                json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            count += 1
            print(f"  [ADDED] {rel_path}: {len(refs)} references")

    print(f"  Total notebooks updated: {count}")


if __name__ == "__main__":
    print("Adding cross-reference cells to notebooks...")
    add_crossref_cells()
    print("Done!")
