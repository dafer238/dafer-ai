"""
Cross-linking script for dafer-ai theory files.

1. Weeks 02-20: convert plain-text "Week XX" references to markdown hyperlinks.
2. Weeks 00a/00b/01: add #section-anchor to existing hyperlinks.
"""

import re, os, unicodedata
from pathlib import Path

BASE = Path(__file__).parent.resolve()

# ── Week number → directory (relative to BASE) ─────────────────────────────

WEEK_DIR = {
    "00a": "01_intro/week00_ai_landscape",
    "00b": "01_intro/week00b_math_and_data",
    "00": "01_intro/week00_ai_landscape",
    "01": "02_fundamentals/week01_optimization",
    "02": "02_fundamentals/week02_advanced_optimizers",
    "03": "02_fundamentals/week03_linear_models",
    "04": "02_fundamentals/week04_dimensionality_reduction",
    "05": "02_fundamentals/week05_clustering",
    "06": "02_fundamentals/week06_regularization",
    "07": "03_probability/week07_likelihood",
    "08": "03_probability/week08_uncertainty",
    "09": "03_probability/week09_time_series",
    "10": "03_probability/week10_surrogate_models",
    "11": "04_neural_networks/week11_nn_from_scratch",
    "12": "04_neural_networks/week12_training_pathologies",
    "13": "05_deep_learning/week13_pytorch_basics",
    "14": "05_deep_learning/week14_training_at_scale",
    "15": "05_deep_learning/week15_cnn_representations",
    "16": "05_deep_learning/week16_regularization_dl",
    "17": "06_sequence_models/week17_attention",
    "18": "06_sequence_models/week18_transformers",
    "19": "07_transfer_learning/week19_finetuning",
    "20": "08_deployment/week20_deployment",
}


def _rel(src: Path, week: str) -> str:
    """Relative path from src's directory to the target week's theory.md."""
    target = BASE / WEEK_DIR[week] / "theory.md"
    return os.path.relpath(target, src.parent).replace("\\", "/")


def _self_week(src: Path) -> str | None:
    m = re.match(r"week(\d+[ab]?)", src.parent.name)
    return m.group(1) if m else None


# ── Slug function (matches python-markdown toc extension) ───────────────────


def _slug(text: str) -> str:
    v = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    v = re.sub(r"[^\w\s-]", "", v).strip().lower()
    return re.sub(r"[-\s]+", "-", v)


def _parse_headings(filepath: Path) -> list[tuple[str, str]]:
    """Return [(heading_text, slug), ...] for a theory.md file."""
    out = []
    for line in filepath.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^#{2,4}\s+(.+)$", line)
        if m:
            h = m.group(1).strip()
            out.append((h, _slug(h)))
    return out


# ── Build heading index for every week ──────────────────────────────────────

HEADINGS: dict[str, list[tuple[str, str]]] = {}
for _wk, _d in WEEK_DIR.items():
    if _wk == "00":
        continue
    _fp = BASE / _d / "theory.md"
    if _fp.exists():
        HEADINGS[_wk] = _parse_headings(_fp)


# ── Keyword → anchor mapping for section-specific linking ───────────────────
# Used when adding anchors to existing links in weeks 00a/00b/01.
# Keys are checked as substrings of the surrounding context (case-insensitive).
# Order matters: first match wins, so put more specific patterns first.

CONCEPT_ANCHOR: dict[str, list[tuple[str, str]]] = {
    "01": [
        ("Section 5.3", "53-variance-and-the-noisespeed-trade-off"),
        ("Section 4.3", "43-convergence-analysis-for-quadratic-losses"),
        ("Section 3.7", "45-full-batch-gradient-descent-in-matrix-form"),
        ("notebook", "11-notebook-reference-guide"),
        ("starter.ipynb", "11-notebook-reference-guide"),
        ("convergence", "9-convergence-diagnostics"),
        ("loss over epochs", "9-convergence-diagnostics"),
        ("SGD", "5-stochastic-gradient-descent-sgd"),
        ("stochastic", "5-stochastic-gradient-descent-sgd"),
        ("mini-batch", "52-mini-batch-sgd"),
        ("learning rate", "8-learning-rate-selection"),
        ("momentum", "6-momentum"),
        ("loss function", "3-loss-functions"),
        ("loss landscape", "7-the-geometry-of-loss-landscapes"),
        ("saddle", "74-saddle-points-in-high-dimensions"),
        ("gradient descent", "4-gradient-descent"),
        ("gradient", "4-gradient-descent"),
        ("optimis", "2-the-optimisation-problem"),
    ],
    "02": [
        ("AdamW", "81-adamw-decoupled-weight-decay"),
        ("weight decay", "81-adamw-decoupled-weight-decay"),
        ("Adam", "7-adam-combining-momentum-and-adaptivity"),
        ("RMSProp", "6-rmsprop-fixing-adagrads-decay"),
        ("adaptive", "5-adagrad-per-parameter-learning-rates"),
        ("cosine", "93-cosine-annealing"),
        ("warmup", "94-warmup"),
        ("schedule", "9-learning-rate-schedules"),
    ],
    "03": [
        ("Section 4", "4-statistical-interpretation-of-linear-regression"),
        ("Section 5", "5-polynomial-regression-and-feature-expansion"),
        ("Section 3.7", "35-gradient-descent-for-linear-regression"),
        ("OLS", "33-the-normal-equations-closed-form-solution"),
        ("normal equation", "33-the-normal-equations-closed-form-solution"),
        ("polynomial", "5-polynomial-regression-and-feature-expansion"),
        ("bias-variance", "6-the-biasvariance-tradeoff"),
        ("bias\u2013variance", "6-the-biasvariance-tradeoff"),
        ("softmax", "77-multi-class-extension-softmax-regression"),
        ("classification", "7-logistic-regression"),
        ("logistic", "7-logistic-regression"),
        ("sigmoid", "72-the-sigmoid-logistic-function"),
        ("cross-entropy", "74-the-loss-function-binary-cross-entropy"),
        ("linear model", "3-linear-regression"),
        ("linear regression", "3-linear-regression"),
        ("linear", "3-linear-regression"),
    ],
    "04": [
        ("Section 8.1", "81-centering-and-scaling"),
        ("Section 4.4", "43-equivalence-of-the-two-views"),
        ("t-SNE", "9-non-linear-alternatives-t-sne-and-umap"),
        ("UMAP", "9-non-linear-alternatives-t-sne-and-umap"),
        ("whitening", "7-pca-whitening"),
        ("eigendecomposition", "32-eigendecomposition-of-the-covariance-matrix"),
        ("SVD", "5-pca-via-the-singular-value-decomposition"),
        ("singular value", "5-pca-via-the-singular-value-decomposition"),
        ("covariance", "3-covariance-and-correlation"),
        ("scree", "62-the-scree-plot"),
        ("explained variance", "61-explained-variance-ratio"),
        ("curse", "2-the-curse-of-dimensionality"),
        ("PCA", "4-principal-component-analysis-the-optimisation-view"),
        ("dimensionality", "4-principal-component-analysis-the-optimisation-view"),
        ("projection", "4-principal-component-analysis-the-optimisation-view"),
    ],
    "05": [
        ("Section 7", "7-cluster-evaluation"),
        ("EM algorithm", "62-the-em-algorithm-intuition"),
        ("EM", "62-the-em-algorithm-intuition"),
        ("BIC", "64-model-selection-bic-and-aic"),
        ("AIC", "64-model-selection-bic-and-aic"),
        ("GMM", "6-gaussian-mixture-models-soft-clustering"),
        ("Gaussian Mixture", "6-gaussian-mixture-models-soft-clustering"),
        ("DBSCAN", "5-dbscan-density-based-clustering"),
        ("hierarchical", "4-hierarchical-clustering"),
        ("silhouette", "35-choosing-k-elbow-method-and-silhouette-analysis"),
        ("K-Means", "3-k-means-clustering"),
        ("K-means", "3-k-means-clustering"),
        ("k-means", "3-k-means-clustering"),
        ("inertia", "31-the-objective-within-cluster-sum-of-squares"),
        ("cluster", "3-k-means-clustering"),
    ],
    "06": [
        ("Section 3.4", "34-ridge-and-svd-shrinkage-of-singular-values"),
        ("Section 3", "3-ridge-regression-l2-regularisation"),
        ("time-series cross-validation", "7-time-series-cross-validation"),
        ("walk-forward", "72-walk-forward-validation"),
        ("early stopping", "8-early-stopping"),
        ("Elastic Net", "5-elastic-net"),
        ("Bayesian", "35-bayesian-interpretation"),
        ("MAP", "35-bayesian-interpretation"),
        ("Lasso", "4-lasso-regression-l1-regularisation"),
        ("L1", "4-lasso-regression-l1-regularisation"),
        ("Ridge", "3-ridge-regression-l2-regularisation"),
        ("L2", "3-ridge-regression-l2-regularisation"),
        ("cross-validation", "6-cross-validation"),
        ("bias-variance", "21-biasvariance-recap"),
        ("bias\u2013variance", "21-biasvariance-recap"),
        ("sparsity", "42-sparsity-why-lasso-produces-zeros"),
        ("regularis", "2-why-regularisation-the-overfitting-problem-revisited"),
    ],
    "07": [
        ("Section 5.2", "52-bernoulli-noise-binary-cross-entropy"),
        ("Section 4.2", "42-properties-of-the-mle"),
        ("AIC", "10-model-selection-aic-and-bic"),
        ("BIC", "10-model-selection-aic-and-bic"),
        ("MAP", "9-from-mle-to-map-the-bridge-to-regularisation"),
        ("Huber", "74-huber-loss-a-practical-compromise"),
        ("Laplace", "72-laplace-noise-mae-l1-loss"),
        ("robust", "7-robust-regression-alternative-noise-models"),
        ("NLL", "5-the-mleloss-function-connection"),
        ("MSE", "51-gaussian-noise-mse"),
        ("cross-entropy", "52-bernoulli-noise-binary-cross-entropy"),
        ("loss function", "5-the-mleloss-function-connection"),
        ("OLS", "63-mle-for-w-recovering-ols"),
        ("linear regression", "6-mle-for-linear-regression-full-derivation"),
        ("exponential family", "54-the-general-recipe"),
        ("Bayes", "23-bayes-theorem"),
        ("MLE", "4-maximum-likelihood-estimation-mle"),
        ("Maximum Likelihood", "4-maximum-likelihood-estimation-mle"),
        ("likelihood", "3-likelihood-from-data-to-models"),
        ("probabilistic", "2-probability-foundations-the-language-of-uncertainty"),
    ],
    "08": [
        ("Section 8.5", "85-connection-to-ridge-regression"),
        ("calibration", "9-calibration"),
        ("conformal", "103-conformal-prediction-distribution-free"),
        ("prediction interval", "10-prediction-intervals"),
        ("BLR", "8-bayesian-linear-regression"),
        ("Bayesian linear regression", "8-bayesian-linear-regression"),
        ("posterior", "8-bayesian-linear-regression"),
        ("prior", "72-prior-likelihood-posterior"),
        ("Monte Carlo", "6-monte-carlo-methods"),
        ("bootstrap", "5-the-bootstrap"),
        ("Bayesian inference", "7-bayesian-inference"),
        ("Bayesian", "7-bayesian-inference"),
        ("ensemble", "11-ensemble-uncertainty"),
        ("aleatoric", "31-aleatoric-uncertainty-data-noise"),
        ("epistemic", "32-epistemic-uncertainty-model-ignorance"),
        ("uncertainty", "3-two-kinds-of-uncertainty"),
        ("confidence", "4-frequentist-uncertainty-confidence-intervals"),
    ],
    "09": [
        ("SARIMA", "9-seasonal-arima-sarima"),
        ("ARIMA", "8-arma-and-arima"),
        ("autoregressive", "6-autoregressive-models-arp"),
        ("AR(", "6-autoregressive-models-arp"),
        ("MA(", "7-moving-average-models-maq"),
        ("ACF", "4-autocorrelation"),
        ("stationarity", "5-stationarity"),
        ("seasonality", "22-seasonality"),
        ("decomposition", "3-decomposition"),
        ("walk-forward", "101-walk-forward-cross-validation"),
        ("forecast", "10-forecast-evaluation"),
        ("lag", "73-lag-features"),
        ("time series", "2-anatomy-of-a-time-series"),
        ("temporal", "2-anatomy-of-a-time-series"),
    ],
    "10": [
        ("ARD", "72-automatic-relevance-determination-ard"),
        ("RBF", "62-rbf-squared-exponential"),
        ("Matérn", "63-matrn-family"),
        ("Expected Improvement", "92-expected-improvement-ei"),
        ("EI", "92-expected-improvement-ei"),
        ("UCB", "93-upper-confidence-bound-ucb"),
        ("acquisition", "92-expected-improvement-ei"),
        ("Bayesian Optimisation", "9-bayesian-optimisation"),
        ("Bayesian Optimization", "9-bayesian-optimisation"),
        ("BO", "9-bayesian-optimisation"),
        ("hyperparameter", "9-bayesian-optimisation"),
        ("marginal likelihood", "7-hyperparameter-optimisation-via-marginal-likelihood"),
        ("kernel", "6-kernel-functions"),
        ("GP posterior", "5-gp-posterior-regression"),
        ("GP prior", "4-gp-prior"),
        ("Gaussian Process", "3-gaussian-processes-intuition"),
        ("GP", "3-gaussian-processes-intuition"),
        ("surrogate", "2-surrogate-modelling"),
    ],
    "11": [
        ("Section 7", "7-weight-initialisation"),
        ("Xavier", "73-xavier-glorot-initialisation"),
        ("Glorot", "73-xavier-glorot-initialisation"),
        ("He ", "74-he-kaiming-initialisation"),
        ("Kaiming", "74-he-kaiming-initialisation"),
        ("gradient checking", "6-gradient-checking"),
        ("autograd", "55-computational-graph-perspective"),
        ("computational graph", "55-computational-graph-perspective"),
        ("backpropagation", "5-backpropagation"),
        ("backprop", "5-backpropagation"),
        ("backward", "5-backpropagation"),
        ("chain rule", "51-the-chain-rule-single-variable"),
        ("activation", "32-activation-functions"),
        ("ReLU", "32-activation-functions"),
        ("sigmoid", "32-activation-functions"),
        ("initialisation", "7-weight-initialisation"),
        ("initialization", "7-weight-initialisation"),
        ("MLP", "4-fully-connected-networks-mlps"),
        ("fully connected", "4-fully-connected-networks-mlps"),
        ("forward pass", "42-forward-pass-in-matrix-form"),
        ("training loop", "8-the-training-loop"),
        ("neuron", "3-the-neuron"),
        ("neural network", "2-from-linear-models-to-neural-networks"),
        ("nonlinear", "2-from-linear-models-to-neural-networks"),
        ("Universal Approximation", "22-the-universal-approximation-idea"),
    ],
    "12": [
        ("dead ReLU", "52-dead-relu-neurons"),
        ("saturation", "5-activation-saturation-and-dead-neurons"),
        ("gradient clipping", "9-fix-4-gradient-clipping"),
        ("clip", "9-fix-4-gradient-clipping"),
        ("residual connection", "10-fix-5-residual-skip-connections"),
        ("skip connection", "10-fix-5-residual-skip-connections"),
        ("ResNet", "10-fix-5-residual-skip-connections"),
        ("LayerNorm", "8-fix-3-layer-normalisation"),
        ("layer normalisation", "8-fix-3-layer-normalisation"),
        ("layer normalization", "8-fix-3-layer-normalisation"),
        ("BatchNorm", "7-fix-2-batch-normalisation"),
        ("batch normalisation", "7-fix-2-batch-normalisation"),
        ("batch normalization", "7-fix-2-batch-normalisation"),
        ("initialisation", "6-fix-1-proper-initialisation"),
        ("vanishing", "3-vanishing-gradients"),
        ("exploding", "4-exploding-gradients"),
        ("gradient flow", "2-the-gradient-flow-problem"),
        ("training patholog", "1-scope-and-purpose"),
    ],
    "13": [
        ("nn.Module", "4-building-models-with-nnmodule"),
        ("nn.Linear", "43-common-layer-types"),
        ("nn.Conv2d", "43-common-layer-types"),
        ("nn.BatchNorm", "43-common-layer-types"),
        ("nn.init", "43-common-layer-types"),
        ("clip_grad_norm", "63-the-optimiser-api"),
        ("DataLoader", "7-data-loading-dataset-and-dataloader"),
        ("Dataset", "7-data-loading-dataset-and-dataloader"),
        ("checkpoint", "9-checkpointing-save-and-load"),
        ("state_dict", "9-checkpointing-save-and-load"),
        ("GPU", "10-device-management-cpu-and-gpu"),
        ("scheduler", "11-learning-rate-schedulers"),
        ("training loop", "8-the-training-loop"),
        ("loss.backward", "32-forward-and-backward"),
        ("autograd", "3-automatic-differentiation-autograd"),
        ("tensor", "2-tensors"),
        ("PyTorch", "1-scope-and-purpose"),
    ],
    "14": [
        ("mixed precision", "6-mixed-precision-training"),
        ("AMP", "6-mixed-precision-training"),
        ("fp16", "6-mixed-precision-training"),
        ("distributed", "7-distributed-training-overview"),
        ("gradient accumulation", "32-gradient-accumulation"),
        ("linear scaling", "33-the-linear-scaling-rule"),
        ("batch size", "3-batching-strategies"),
        ("checkpoint", "5-robust-checkpointing"),
        ("warmup", "46-warmup"),
        ("cosine", "44-cosine-annealing"),
        ("one-cycle", "48-one-cycle-policy"),
        ("learning rate schedule", "4-learning-rate-schedules"),
        ("DataLoader", "2-the-data-pipeline"),
        ("worker", "22-multi-worker-loading"),
        ("training at scale", "1-scope-and-purpose"),
    ],
    "15": [
        ("feature extraction", "102-feature-extraction-frozen-backbone"),
        ("fine-tuning", "103-fine-tuning-unfrozen-backbone"),
        ("fine-tune", "103-fine-tuning-unfrozen-backbone"),
        ("transfer learning", "10-transfer-learning-and-pretrained-features"),
        ("transfer", "10-transfer-learning-and-pretrained-features"),
        ("receptive field", "6-receptive-fields"),
        ("pooling", "5-pooling"),
        ("Global Average Pooling", "53-global-average-pooling"),
        ("BatchNorm2d", "83-batchnorm2d-in-cnns"),
        ("filter", "91-first-layer-filters"),
        ("activation", "92-forward-hooks-for-activations"),
        ("ablation", "11-ablation-studies"),
        ("ResNet", "12-classic-cnn-architectures"),
        ("VGG", "12-classic-cnn-architectures"),
        ("convolution", "3-the-convolution-operation"),
        ("CNN", "2-from-fully-connected-to-convolutional"),
        ("inductive bias", "22-inductive-biases-of-convolution"),
        ("augmentation", "5-data-augmentation"),
    ],
    "16": [
        ("MC Dropout", "93-dropout-as-approximate-ensemble"),
        ("label smoothing", "10-label-smoothing"),
        ("ensemble", "9-ensemble-methods"),
        ("ablation", "11-regularisation-interactions-and-ablation-protocol"),
        ("data augmentation", "5-data-augmentation"),
        ("augmentation", "5-data-augmentation"),
        ("weight decay", "4-weight-decay-l2-regularisation"),
        ("Dropout", "3-dropout"),
        ("dropout", "3-dropout"),
        ("BatchNorm", "6-batchnorm-as-a-regulariser"),
        ("early stopping", "8-early-stopping"),
        ("overfitting", "2-overfitting-in-deep-networks"),
        ("regularis", "1-scope-and-purpose"),
    ],
    "17": [
        ("Bahdanau", "82-additive-bahdanau-attention"),
        ("masking", "44-masking"),
        ("causal", "44-masking"),
        ("multi-head", "6-multi-head-attention"),
        ("Multi-Head", "6-multi-head-attention"),
        ("self-attention", "7-self-attention-vs-cross-attention"),
        ("cross-attention", "7-self-attention-vs-cross-attention"),
        ("seq2seq", "8-attention-in-sequence-to-sequence-models"),
        ("encoder-decoder", "8-attention-in-sequence-to-sequence-models"),
        ("QK", "41-the-formula"),
        ("scaled dot-product", "4-scaled-dot-product-attention"),
        ("score", "5-scoring-functions"),
        ("attention", "3-attention-as-soft-lookup"),
    ],
    "18": [
        ("Pre-LN", "45-post-ln-vs-pre-ln"),
        ("Post-LN", "45-post-ln-vs-pre-ln"),
        ("scaling law", "83-scaling-laws"),
        ("warm-up", "91-learning-rate-warm-up"),
        ("warmup", "91-learning-rate-warm-up"),
        ("dropout", "92-dropout-placement"),
        ("FFN", "42-sub-layer-2-position-wise-feed-forward-network"),
        ("feed-forward", "42-sub-layer-2-position-wise-feed-forward-network"),
        ("decoder", "6-the-transformer-decoder"),
        ("encoder layer", "4-the-transformer-encoder-layer"),
        ("residual", "43-residual-connections"),
        ("LayerNorm", "44-layer-normalisation"),
        ("positional", "3-input-representation"),
        ("embedding", "31-token-embeddings"),
        ("autoregressive", "7-encoderdecoder-vs-encoder-only-vs-decoder-only"),
        ("transformer", "2-the-transformer-at-a-glance"),
        ("Transformer", "2-the-transformer-at-a-glance"),
    ],
    "19": [
        ("catastrophic forgetting", "9-catastrophic-forgetting"),
        ("LoRA", "52-lora-low-rank-adaptation"),
        ("adapter", "51-adapters"),
        ("prefix tuning", "53-prefix-tuning-and-prompt-tuning"),
        ("prompt tuning", "53-prefix-tuning-and-prompt-tuning"),
        ("PEFT", "5-parameter-efficient-fine-tuning-peft"),
        ("discriminative", "43-discriminative-learning-rates"),
        ("HuggingFace", "7-fine-tuning-pretrained-transformers-huggingface-workflow"),
        ("feature extraction", "31-feature-extraction-frozen-backbone"),
        ("freezing", "31-feature-extraction-frozen-backbone"),
        ("frozen", "31-feature-extraction-frozen-backbone"),
        ("gradual unfreezing", "33-partial-fine-tuning-gradual-unfreezing"),
        ("full fine-tuning", "32-full-fine-tuning"),
        ("fine-tun", "3-adaptation-strategies"),
        ("transfer", "2-transfer-learning-the-core-idea"),
    ],
    "20": [
        ("TorchScript", "55-torchscript"),
        ("ONNX", "54-onnx-export-and-runtime"),
        ("distillation", "53-knowledge-distillation"),
        ("pruning", "52-pruning"),
        ("quantisation", "51-quantisation"),
        ("quantization", "51-quantisation"),
        ("Docker", "7-containerisation-with-docker"),
        ("container", "7-containerisation-with-docker"),
        ("FastAPI", "6-building-an-inference-api"),
        ("API", "6-building-an-inference-api"),
        ("monitoring", "8-health-checks-and-monitoring"),
        ("health", "8-health-checks-and-monitoring"),
        ("drift", "10-data-and-model-drift"),
        ("CI/CD", "11-cicd-for-ml"),
        ("serialisation", "31-model-serialisation"),
        ("inference", "4-inference-constraints"),
        ("latency", "41-latency"),
        ("deployment", "3-from-notebook-to-production"),
        ("calibration", "8-health-checks-and-monitoring"),
    ],
}


def _best_anchor(context: str, target_week: str) -> str:
    """Return '#slug' for the best matching section, or '' if no confident match."""
    kw_list = CONCEPT_ANCHOR.get(target_week, [])
    ctx_lower = context.lower()
    for keyword, slug in kw_list:
        if keyword.lower() in ctx_lower:
            return "#" + slug
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Add links to plain-text "Week XX" in weeks 02-20
# ══════════════════════════════════════════════════════════════════════════════

THEORY_02_20 = [BASE / WEEK_DIR[str(w).zfill(2)] / "theory.md" for w in range(2, 21)]

# Regex for ranges: "Week(s) XX–YY" or "Week(s) XX-YY" or "Week XX/YY"
_RANGE_RE = re.compile(r"(Weeks?\s)(0[0-9][ab]?|[12][0-9])([–\-/])(0[0-9][ab]?|[12][0-9])")
# Regex for individual: "Week(s) XX"
_SINGLE_RE = re.compile(r"(Weeks?\s)(0[0-9][ab]?|[12][0-9])(?![0-9ab])")
# Regex to protect existing markdown links
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")


def _linkify_line(line: str, src: Path, self_wk: str | None) -> str:
    """Convert plain-text Week references to markdown links."""
    # Protect existing markdown links
    protected = {}
    ctr = [0]

    def _prot(m):
        key = f"ZZL{ctr[0]}ZZ"
        ctr[0] += 1
        protected[key] = m.group(0)
        return key

    line = _LINK_RE.sub(_prot, line)

    # Replace ranges first
    def _repl_range(m):
        prefix, n1, sep, n2 = m.group(1), m.group(2), m.group(3), m.group(4)
        parts = []
        if n1 != self_wk and n1 in WEEK_DIR:
            parts.append(f"[{prefix}{n1}]({_rel(src, n1)})")
        else:
            parts.append(f"{prefix}{n1}")
        parts.append(sep)
        if n2 != self_wk and n2 in WEEK_DIR:
            parts.append(f"[{n2}]({_rel(src, n2)})")
        else:
            parts.append(n2)
        return "".join(parts)

    line = _RANGE_RE.sub(_repl_range, line)

    # Replace individual references
    def _repl_single(m):
        prefix, num = m.group(1), m.group(2)
        if num == self_wk or num not in WEEK_DIR:
            return m.group(0)
        return f"[{prefix}{num}]({_rel(src, num)})"

    line = _SINGLE_RE.sub(_repl_single, line)

    # Restore protected links
    for k, v in protected.items():
        line = line.replace(k, v)
    return line


def add_links_weeks_02_20():
    total = 0
    for fp in THEORY_02_20:
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        sw = _self_week(fp)
        lines = text.split("\n")
        new_lines = []
        in_code = False
        changed = 0
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
            if in_code:
                new_lines.append(line)
                continue
            new_line = _linkify_line(line, fp, sw)
            if new_line != line:
                changed += 1
            new_lines.append(new_line)
        new_text = "\n".join(new_lines)
        if new_text != text:
            fp.write_text(new_text, encoding="utf-8")
            print(f"  [LINKED] {fp.relative_to(BASE)}: {changed} lines changed")
            total += changed
        else:
            print(f"  [SKIP]   {fp.relative_to(BASE)}: no changes")
    print(f"  Total lines changed in weeks 02-20: {total}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Add #anchors to existing links in weeks 00a/00b/01
# ══════════════════════════════════════════════════════════════════════════════

THEORY_00_01 = [
    BASE / "01_intro/week00_ai_landscape/theory.md",
    BASE / "01_intro/week00b_math_and_data/theory.md",
    BASE / "02_fundamentals/week01_optimization/theory.md",
]

# Regex to find existing cross-links: [text](path/theory.md)
_EXISTING_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+/theory\.md)\)")


def _resolve_target_week(href: str) -> str | None:
    """Given a relative href like ../../02_fundamentals/.../theory.md, find the week number."""
    for wk, dirpath in WEEK_DIR.items():
        if wk == "00":
            continue
        # Check if the href ends with this week's path
        target_suffix = dirpath.replace("/", "/") + "/theory.md"
        if href.replace("\\", "/").endswith(target_suffix):
            return wk
        # Also match just the week folder
        week_folder = dirpath.split("/")[-1]  # e.g. "week01_optimization"
        if week_folder + "/theory.md" in href:
            return wk
    return None


def add_anchors_weeks_00_01():
    total = 0
    for fp in THEORY_00_01:
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        lines = text.split("\n")
        new_lines = []
        changed = 0

        for line in lines:
            # For each existing cross-link on this line, add an anchor
            def _add_anchor(m):
                link_text = m.group(1)
                href = m.group(2)
                # Already has an anchor?
                if "#" in href:
                    return m.group(0)
                target_wk = _resolve_target_week(href)
                if target_wk is None:
                    return m.group(0)
                # Get surrounding context (the whole line)
                anchor = _best_anchor(line, target_wk)
                if anchor:
                    return f"[{link_text}]({href}{anchor})"
                return m.group(0)

            new_line = _EXISTING_LINK_RE.sub(_add_anchor, line)
            if new_line != line:
                changed += 1
            new_lines.append(new_line)

        new_text = "\n".join(new_lines)
        if new_text != text:
            fp.write_text(new_text, encoding="utf-8")
            print(f"  [ANCHOR] {fp.relative_to(BASE)}: {changed} lines changed")
            total += changed
        else:
            print(f"  [SKIP]   {fp.relative_to(BASE)}: no changes")
    print(f"  Total lines changed in weeks 00a/00b/01: {total}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: Adding hyperlinks to weeks 02-20")
    print("=" * 60)
    add_links_weeks_02_20()
    print()
    print("=" * 60)
    print("PART 2: Adding section anchors to weeks 00a/00b/01")
    print("=" * 60)
    add_anchors_weeks_00_01()
    print()
    print("Done!")
