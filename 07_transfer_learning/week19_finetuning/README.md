Week 15 — Fine-Tuning & Transfer Learning

## Prerequisites

- **Week 14** — transformers (for NLP/sequence transfer); or **Week 11** (for vision transfer via CNNs).
- **Week 04** — regularization. Freezing layers during fine-tuning is a form of parameter regularization.

## What this week delivers

Most practical AI work does not train from scratch. It starts from a pretrained model and adapts. Understanding _when_ and _how_ to fine-tune — and when feature extraction is better — is the practical skill that separates practitioners from users.

Overview
Study transfer learning strategies, when to fine-tune vs use feature extractors, and lightweight methods like adapters.

Study
- Transfer learning paradigms; domain gap and covariate shift
- Adapters and parameter-efficient fine-tuning

Practical libraries & tools
- HuggingFace Transformers for NLP; PyTorch for vision transfer

Datasets & examples
- Small target-domain datasets; use pretrained checkpoints and adapt

Exercises
1) Fine-tune a pretrained model on a small downstream task and report metrics.

2) Compare feature-extraction vs full fine-tuning across limited-data regimes.

3) Try adapter or partial-freeze strategies and compare parameter efficiency.

Deliverable
- Notebook with fine-tuning experiments, lessons learned, and suggested workflow for your capstone.

## What comes next

- **Week 16** (Deployment & Capstone) — the model you fine-tune this week becomes the candidate for deployment.
