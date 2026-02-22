"""
Fix notebook title cells: update the week numbers in title markdown cells
to match the folder/directory week numbers.
Also add a cross-reference markdown cell after the title cell.
"""

import json
from pathlib import Path

BASE = Path(__file__).parent.resolve()

# (folder_path, wrong_title_prefix, correct_title, theory_link)
# theory_link is relative from the notebook's directory to theory.md
FIXES = [
    # week04: title says "Week 03" → should be "Week 04"
    (
        "02_fundamentals/week04_dimensionality_reduction/starter.ipynb",
        "# Week 03",
        "# Week 04 — Dimensionality Reduction & PCA",
        "theory.md",
    ),
    # week05: title says "Week 03" → "Week 05"
    (
        "02_fundamentals/week05_clustering/starter.ipynb",
        "# Week 03",
        "# Week 05 — Clustering",
        "theory.md",
    ),
    # week06: title says "Week 04" → "Week 06"
    (
        "02_fundamentals/week06_regularization/starter.ipynb",
        "# Week 04",
        "# Week 06 — Regularization & Validation",
        "theory.md",
    ),
    # week07: title says "Week 05" → "Week 07"
    (
        "03_probability/week07_likelihood/starter.ipynb",
        "# Week 05",
        "# Week 07 — Probability & Noise (Likelihood)",
        "theory.md",
    ),
    # week08: title says "Week 06" → "Week 08"
    (
        "03_probability/week08_uncertainty/starter.ipynb",
        "# Week 06",
        "# Week 08 — Uncertainty & Statistics",
        "theory.md",
    ),
    # week09: title says "Week 07" → "Week 09"
    (
        "03_probability/week09_time_series/starter.ipynb",
        "# Week 07",
        "# Week 09 — Time-Series Fundamentals: Seasonality & Autocorrelation",
        "theory.md",
    ),
    # week10: title says "Week 07" → "Week 10"
    (
        "03_probability/week10_surrogate_models/starter.ipynb",
        "# Week 07",
        "# Week 10 — Surrogate Models & Gaussian Processes",
        "theory.md",
    ),
    # week11: title says "Week 07" → "Week 11"
    (
        "04_neural_networks/week11_nn_from_scratch/starter.ipynb",
        "# Week 07",
        "# Week 11 — Neural Networks From Scratch",
        "theory.md",
    ),
    # week12: title says "Week 08" → "Week 12"
    (
        "04_neural_networks/week12_training_pathologies/starter.ipynb",
        "# Week 08",
        "# Week 12 — Training Pathologies",
        "theory.md",
    ),
    # week13: title says "Week 09" → "Week 13"
    (
        "05_deep_learning/week13_pytorch_basics/starter.ipynb",
        "# Week 09",
        "# Week 13 — PyTorch Fundamentals",
        "theory.md",
    ),
    # week14: title says "Week 10" → "Week 14"
    (
        "05_deep_learning/week14_training_at_scale/starter.ipynb",
        "# Week 10",
        "# Week 14 — Efficient Training (Training at Scale)",
        "theory.md",
    ),
    # week15: title says "Week 11" → "Week 15"
    (
        "05_deep_learning/week15_cnn_representations/starter.ipynb",
        "# Week 11",
        "# Week 15 — Representation Learning (CNNs)",
        "theory.md",
    ),
    # week16: title says "Week 12" → "Week 16"
    (
        "05_deep_learning/week16_regularization_dl/starter.ipynb",
        "# Week 12",
        "# Week 16 — Regularization at Scale",
        "theory.md",
    ),
    # week17: title says "Week 13" → "Week 17"
    (
        "06_sequence_models/week17_attention/starter.ipynb",
        "# Week 13",
        "# Week 17 — Attention Mechanisms",
        "theory.md",
    ),
    # week18: title says "Week 14" → "Week 18"
    (
        "06_sequence_models/week18_transformers/starter.ipynb",
        "# Week 14",
        "# Week 18 — Transformers",
        "theory.md",
    ),
    # week19: title says "Week 15" → "Week 19"
    (
        "07_transfer_learning/week19_finetuning/starter.ipynb",
        "# Week 15",
        "# Week 19 — Fine-Tuning & Transfer Learning",
        "theory.md",
    ),
    # week20: title says "Week 16" → "Week 20"
    (
        "08_deployment/week20_deployment/starter.ipynb",
        "# Week 16",
        "# Week 20 — Deployment & Capstone",
        "theory.md",
    ),
]


def fix_titles():
    for rel_path, wrong_prefix, correct_title, _ in FIXES:
        nb_path = BASE / rel_path
        if not nb_path.exists():
            print(f"  [SKIP] {rel_path} not found")
            continue

        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])

        # Find the first markdown cell (title cell)
        fixed = False
        for cell in cells:
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                if source and source[0].startswith(wrong_prefix):
                    # Replace the title line
                    rest = source[0].split("\n", 1)
                    if len(rest) > 1:
                        source[0] = correct_title + "\n" + rest[1]
                    else:
                        source[0] = correct_title
                    cell["source"] = source
                    fixed = True
                    print(f"  [FIXED] {rel_path}: {wrong_prefix}... → {correct_title}")
                break

        if not fixed:
            print(f"  [WARN]  {rel_path}: title cell not matched (expected '{wrong_prefix}')")
            continue

        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print("Fixing notebook titles...")
    fix_titles()
    print("Done!")
