"""Fix ToC indentation: change 3-space indent to 4-space inside ## Table of Contents sections.

Python-markdown requires 4 spaces for nested lists under ordered lists.
Affects lines matching: ^   - \d   (3 spaces + dash + digit = sub-item)
Only within the ToC block (between '## Table of Contents' and '---' or next '##').
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

FILES = [
    "01_intro/week00b_math_and_data/theory.md",
    "02_fundamentals/week01_optimization/theory.md",
    "02_fundamentals/week02_advanced_optimizers/theory.md",
    "02_fundamentals/week05_clustering/theory.md",
    "02_fundamentals/week06_regularization/theory.md",
    "03_probability/week08_uncertainty/theory.md",
    "03_probability/week09_time_series/theory.md",
    "03_probability/week10_surrogate_models/theory.md",
    "04_neural_networks/week11_nn_from_scratch/theory.md",
    "04_neural_networks/week12_training_pathologies/theory.md",
    "05_deep_learning/week13_pytorch_basics/theory.md",
    "05_deep_learning/week14_training_at_scale/theory.md",
    "05_deep_learning/week15_cnn_representations/theory.md",
    "05_deep_learning/week16_regularization_dl/theory.md",
    "06_sequence_models/week17_attention/theory.md",
    "06_sequence_models/week18_transformers/theory.md",
    "07_transfer_learning/week19_finetuning/theory.md",
    "08_deployment/week20_deployment/theory.md",
]

total_fixes = 0

for rel in FILES:
    path = ROOT / rel
    if not path.exists():
        print(f"  SKIP (not found): {rel}")
        continue

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_toc = False
    fixes = 0

    for i, line in enumerate(lines):
        stripped = line.rstrip("\n\r")

        # Enter ToC block
        if stripped.strip() == "## Table of Contents":
            in_toc = True
            continue

        # Exit ToC block at next heading or horizontal rule
        if in_toc and (stripped.startswith("## ") or stripped.strip() == "---"):
            in_toc = False
            continue

        # Fix 3-space-indented sub-items (  `   - X.Y ...` -> `    - X.Y ...`)
        if in_toc and re.match(r"^   - \d", line):
            lines[i] = " " + line  # add one space at the front
            fixes += 1

    if fixes:
        path.write_text("".join(lines), encoding="utf-8")
        print(f"  {rel}: {fixes} lines fixed")
        total_fixes += fixes
    else:
        print(f"  {rel}: OK (no 3-space items found)")

print(f"\nTotal: {total_fixes} lines fixed across {len(FILES)} files")
