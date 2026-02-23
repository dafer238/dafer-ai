# Deployment and ML Lifecycle

## Table of Contents

1. [Scope and Purpose](#1-scope-and-purpose)
2. [The ML Lifecycle](#2-the-ml-lifecycle)
3. [From Notebook to Production](#3-from-notebook-to-production)
    - 3.1 [Model Serialisation](#31-model-serialisation)
    - 3.2 [Inference vs. Training Mode](#32-inference-vs-training-mode)
    - 3.3 [Reproducibility Checklist](#33-reproducibility-checklist)
4. [Inference Constraints](#4-inference-constraints)
    - 4.1 [Latency](#41-latency)
    - 4.2 [Throughput](#42-throughput)
    - 4.3 [Memory](#43-memory)
    - 4.4 [The Latency–Throughput Trade-off](#44-the-latencythroughput-trade-off)
5. [Model Optimisation for Deployment](#5-model-optimisation-for-deployment)
    - 5.1 [Quantisation](#51-quantisation)
    - 5.2 [Pruning](#52-pruning)
    - 5.3 [Knowledge Distillation](#53-knowledge-distillation)
    - 5.4 [ONNX Export and Runtime](#54-onnx-export-and-runtime)
    - 5.5 [TorchScript](#55-torchscript)
6. [Building an Inference API](#6-building-an-inference-api)
    - 6.1 [FastAPI Fundamentals](#61-fastapi-fundamentals)
    - 6.2 [Request/Response Schema with Pydantic](#62-requestresponse-schema-with-pydantic)
    - 6.3 [Model Loading and Lifecycle](#63-model-loading-and-lifecycle)
    - 6.4 [A Complete Prediction Endpoint](#64-a-complete-prediction-endpoint)
    - 6.5 [Running with Uvicorn](#65-running-with-uvicorn)
7. [Containerisation with Docker](#7-containerisation-with-docker)
    - 7.1 [Why Containers?](#71-why-containers)
    - 7.2 [Dockerfile Anatomy](#72-dockerfile-anatomy)
    - 7.3 [Building and Running](#73-building-and-running)
    - 7.4 [Best Practices](#74-best-practices)
8. [Health Checks and Monitoring](#8-health-checks-and-monitoring)
    - 8.1 [Health Endpoints](#81-health-endpoints)
    - 8.2 [Request Latency Logging](#82-request-latency-logging)
    - 8.3 [Prometheus Metrics](#83-prometheus-metrics)
    - 8.4 [What to Monitor](#84-what-to-monitor)
9. [Model Serving Frameworks](#9-model-serving-frameworks)
10. [Data and Model Drift](#10-data-and-model-drift)
11. [CI/CD for ML](#11-cicd-for-ml)
12. [Security and Validation](#12-security-and-validation)
13. [Connections to the Rest of the Course](#13-connections-to-the-rest-of-the-course)
14. [Notebook Reference Guide](#14-notebook-reference-guide)
15. [Symbol Reference](#15-symbol-reference)
16. [References](#16-references)

---

## 1. Scope and Purpose

This week completes the ML lifecycle: **data → model → evaluate → deploy → monitor**. Every previous week built components in isolation. This week connects them into a working system.

After this week you will be able to:

1. **Serialise and load** a trained PyTorch model for inference.
2. **Build a minimal REST API** with FastAPI that serves predictions.
3. **Containerise** the API with Docker for reproducible deployment.
4. **Add health checks and latency monitoring** to a serving endpoint.
5. **Reason about** inference constraints (latency, throughput, memory) and model optimisation for deployment.

**Prerequisites.** All previous weeks — you need a trained, evaluated model. [Week 14](../../05_deep_learning/week14_training_at_scale/theory.md#5-robust-checkpointing) (training at scale — checkpointing, mixed precision). [Week 19](../../07_transfer_learning/week19_finetuning/theory.md#3-adaptation-strategies) (fine-tuning — the model you deploy).

---

## 2. The ML Lifecycle

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   Data    │───►│  Train   │───►│ Evaluate │───►│  Deploy  │───►│ Monitor  │
│ Pipeline  │    │  Model   │    │  & Test  │    │  & Serve │    │ & Retrain│
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └─────┬────┘
     ▲                                                                │
     └────────────────────────────────────────────────────────────────┘
                              feedback loop
```

| Stage          | Weeks covered        | Key outputs                                  |
| -------------- | -------------------- | -------------------------------------------- |
| Data pipeline  | [Week 00a](../../01_intro/week00_ai_landscape/theory.md)–[Week 00b](../../01_intro/week00b_math_and_data/theory.md), 14    | Clean, versioned datasets; DataLoaders       |
| Model training | [Week 01](../../02_fundamentals/week01_optimization/theory.md#4-gradient-descent)–[Week 18](../../06_sequence_models/week18_transformers/theory.md#2-the-transformer-at-a-glance)          | Trained weights (`.pt`, `.safetensors`)      |
| Evaluation     | [Weeks 03](../../02_fundamentals/week03_linear_models/theory.md), 06, 08, 16 | Metrics, calibration, uncertainty estimates  |
| Deployment     | **This week**        | API, container, serving infrastructure       |
| Monitoring     | **This week**        | Health checks, latency logs, drift detection |

The lifecycle is **iterative** — monitoring reveals model degradation, which triggers retraining with new data.

---

## 3. From Notebook to Production

### 3.1 Model Serialisation

**Option 1: Save state_dict (recommended)**

```python
# Save
torch.save(model.state_dict(), 'model_weights.pt')

# Load
model = MyModelClass(...)   # instantiate the architecture
model.load_state_dict(torch.load('model_weights.pt', weights_only=True))
model.eval()
```

This saves only the learned parameters — the architecture is defined in code. This is more robust to code changes and more portable.

**Option 2: Save entire model (pickle)**

```python
torch.save(model, 'model.pt')
model = torch.load('model.pt')
```

This pickles the entire object, including the class definition. Fragile — it breaks if you rename or move the class.

**Option 3: SafeTensors (HuggingFace)**

```python
from safetensors.torch import save_model, load_model
save_model(model, 'model.safetensors')
load_model(model, 'model.safetensors')
```

Safer than pickle (no arbitrary code execution), faster loading, and supports memory-mapped loading for large models.

| Method       | Portable | Safe     | Fast | Recommended for          |
| ------------ | -------- | -------- | ---- | ------------------------ |
| `state_dict` | Yes      | Moderate | Good | General PyTorch          |
| Full pickle  | No       | No       | Good | Quick experiments        |
| SafeTensors  | Yes      | Yes      | Best | Production, large models |

---

### 3.2 Inference vs. Training Mode

Before serving predictions, the model **must** be in eval mode:

```python
model.eval()                    # disables dropout, sets BatchNorm to use running stats
with torch.no_grad():           # disables gradient computation → saves memory
    output = model(input_tensor)
```

| Behaviour         | `model.train()`         | `model.eval()`                       |
| ----------------- | ----------------------- | ------------------------------------ |
| Dropout           | Active (random zeroing) | Disabled                             |
| BatchNorm         | Uses mini-batch stats   | Uses running mean/variance           |
| Gradient tracking | Enabled (default)       | Enabled (use `no_grad()` to disable) |

Forgetting `model.eval()` causes **non-deterministic predictions** (dropout is still active) and degraded accuracy (BatchNorm uses batch statistics).

---

### 3.3 Reproducibility Checklist

Before deployment, ensure:

- [ ] Model architecture code is versioned (git).
- [ ] Training hyperparameters are logged (config file, W&B, MLflow).
- [ ] Random seeds are set (Python, NumPy, PyTorch, CUDA).
- [ ] Dataset version is tracked (hash, DVC, or version tag).
- [ ] Dependencies are pinned (`requirements.txt` with exact versions).
- [ ] Evaluation metrics are recorded and reproducible.
- [ ] The model file is the exact checkpoint used for evaluation.

---

## 4. Inference Constraints

Production systems have constraints that training does not.

### 4.1 Latency

**Definition.** Time from request arrival to response delivery.

$$\text{Latency} = t_\text{preprocess} + t_\text{inference} + t_\text{postprocess} + t_\text{network}$$

Typical budgets:

| Application                 | Latency budget  |
| --------------------------- | --------------- |
| Real-time serving (web API) | < 100 ms        |
| Interactive chat            | < 500 ms        |
| Batch scoring               | Seconds–minutes |
| Offline analytics           | Hours           |

**Key insight:** model inference is often not the bottleneck. Data preprocessing (tokenisation, normalisation) and serialisation (JSON parsing, tensor conversion) can dominate.

---

### 4.2 Throughput

**Definition.** Number of requests handled per unit time.

$$\text{Throughput} = \frac{\text{requests}}{\text{second}} \quad (\text{RPS or QPS})$$

Throughput scales with:
- **Batching:** processing multiple inputs at once (GPU utilisation).
- **Concurrency:** handling multiple requests in parallel (async workers).
- **Hardware:** faster GPU, more CPU cores, more memory bandwidth.

---

### 4.3 Memory

A model must fit in memory (GPU or CPU) at serving time:

$$\text{Memory} \approx \underbrace{4 \times N_\text{params}}_{\text{FP32 weights}} + \underbrace{\text{activation memory}}_{\text{depends on batch, seq\_len}}$$

| Model       | Parameters | FP32 memory | FP16 memory | INT8 memory |
| ----------- | ---------- | ----------- | ----------- | ----------- |
| DistilBERT  | 66M        | 264 MB      | 132 MB      | 66 MB       |
| BERT-base   | 110M       | 440 MB      | 220 MB      | 110 MB      |
| GPT-2 Small | 117M       | 468 MB      | 234 MB      | 117 MB      |
| LLaMA-7B    | 7B         | 28 GB       | 14 GB       | 7 GB        |

Activation memory grows with batch size and sequence length. For serving, batch size is typically small (1–8), so weight memory dominates.

---

### 4.4 The Latency–Throughput Trade-off

| Strategy                                             | Latency                 | Throughput | When to use           |
| ---------------------------------------------------- | ----------------------- | ---------- | --------------------- |
| Process one request at a time                        | Lowest (no wait)        | Low        | Real-time, low volume |
| Dynamic batching (buffer requests, process together) | Higher (wait for batch) | High       | Medium–high volume    |
| Continuous batching                                  | Moderate                | Highest    | LLM serving (vLLM)    |

Dynamic batching collects requests for up to $t_\text{wait}$ ms, then processes them as a single batch:

$$\text{Effective latency} = t_\text{wait} + t_\text{inference}(\text{batch\_size})$$

Since GPU inference time grows sub-linearly with batch size (parallelism), batching improves throughput at the cost of individual latency.

---

## 5. Model Optimisation for Deployment

### 5.1 Quantisation

Reduce numerical precision to shrink model size and speed up inference:

$$W_\text{float32} \xrightarrow{\text{quantise}} W_\text{int8} \quad \text{(4× smaller, 2–4× faster)}$$

**Types:**

| Type                                  | When              | How                                              |
| ------------------------------------- | ----------------- | ------------------------------------------------ |
| **Post-training quantisation (PTQ)**  | After training    | Calibrate with a small dataset; no retraining    |
| **Quantisation-aware training (QAT)** | During training   | Simulate quantisation in forward pass; fine-tune |
| **Dynamic quantisation**              | At inference time | Weights are quantised; activations stay FP32     |

```python
# PyTorch dynamic quantisation (CPU only)
import torch.quantization
quantised_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

| Precision | Bits | Memory ratio | Accuracy loss                |
| --------- | ---- | ------------ | ---------------------------- |
| FP32      | 32   | 1×           | Baseline                     |
| FP16      | 16   | 0.5×         | Negligible                   |
| INT8      | 8    | 0.25×        | Usually < 1%                 |
| INT4      | 4    | 0.125×       | 1–3% (with good calibration) |

---

### 5.2 Pruning

Remove weights (set to zero) that contribute little to predictions:

$$W'_{ij} = \begin{cases}W_{ij} & \text{if } |W_{ij}| > \tau\\0 & \text{otherwise}\end{cases}$$

**Unstructured pruning** zeroes individual weights (sparse matrix — hard to accelerate without specialised hardware). **Structured pruning** removes entire neurons, attention heads, or layers (directly reduces computation).

Typical result: 50–80% of weights can be pruned with < 1% accuracy loss, but real speedup requires structured pruning or sparse-aware hardware.

---

### 5.3 Knowledge Distillation

Train a small "student" model to mimic a large "teacher" model:

$$\mathcal{L}_\text{distill} = (1 - \alpha)\,\mathcal{L}_\text{CE}(y, \hat{y}_\text{student}) + \alpha\,T^2\,\text{KL}(\hat{p}_\text{teacher}^{(T)} \| \hat{p}_\text{student}^{(T)})$$

where:
- $\hat{p}^{(T)} = \text{softmax}(z / T)$ are "soft" predictions with temperature $T > 1$.
- $\alpha$ balances the hard label loss and the distillation loss.
- $T^2$ compensates for the reduced gradient magnitude at high temperature.

**Intuition.** Soft predictions contain more information than hard labels — they encode relative class similarities ("this cat image is slightly dog-like"). The student learns from these "dark knowledge" signals.

Example: DistilBERT is BERT distilled — 40% smaller, 60% faster, retains 97% of BERT's accuracy.

---

### 5.4 ONNX Export and Runtime

ONNX (Open Neural Network Exchange) is a vendor-neutral format:

```python
import torch.onnx

dummy_input = torch.randint(0, 1000, (1, 128))
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'}}
)
```

ONNX Runtime provides optimised inference with graph fusion, quantisation, and hardware-specific backends:

```python
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
output = session.run(None, {'input_ids': input_array})
```

Typical speedup: 1.5–3× over PyTorch eager mode, depending on model and hardware.

---

### 5.5 TorchScript

PyTorch's built-in JIT compiler for deployment:

```python
# Tracing (follows one execution path)
traced = torch.jit.trace(model, example_input)

# Scripting (analyses Python code)
scripted = torch.jit.script(model)

# Save for C++ / non-Python deployment
traced.save('model_traced.pt')
```

TorchScript removes the Python dependency, enabling deployment in C++, mobile (PyTorch Mobile), or edge devices.

---

## 6. Building an Inference API

### 6.1 FastAPI Fundamentals

FastAPI is a modern Python web framework built on ASGI (Asynchronous Server Gateway Interface):

- **Type-annotated** — automatic request/response validation.
- **Async-native** — supports `async def` for non-blocking I/O.
- **Auto-docs** — Swagger UI at `/docs`, OpenAPI schema at `/openapi.json`.
- **Fast** — comparable to Go/Node.js for I/O-bound workloads.

```python
from fastapi import FastAPI

app = FastAPI(title="ML Model API", version="1.0.0")
```

---

### 6.2 Request/Response Schema with Pydantic

Define typed schemas for input validation:

```python
from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    text: str = Field(..., description="Input text for classification")
    
class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
```

Pydantic validates incoming JSON automatically — malformed requests return a 422 error with details.

---

### 6.3 Model Loading and Lifecycle

Load the model **once at startup**, not per request:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    app.state.model = load_model('model_weights.pt')
    app.state.model.eval()
    app.state.tokenizer = load_tokenizer('distilbert-base-uncased')
    yield
    # Shutdown: cleanup
    del app.state.model

app = FastAPI(lifespan=lifespan)
```

**Why at startup?**
- Model loading is slow (seconds for large models).
- Loading per request would add seconds of latency.
- The model is read-only during inference — safe to share across requests.

---

### 6.4 A Complete Prediction Endpoint

```python
import torch
import time

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = time.time()
    
    # 1. Preprocess
    encoded = app.state.tokenizer(
        request.text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # 2. Inference
    with torch.no_grad():
        logits = app.state.model(**encoded).logits
    
    # 3. Postprocess
    probs = torch.softmax(logits, dim=-1)
    confidence, label_idx = probs.max(dim=-1)
    label = ['negative', 'positive'][label_idx.item()]
    
    latency = time.time() - start
    print(f"Prediction latency: {latency*1000:.1f}ms")
    
    return PredictionResponse(label=label, confidence=confidence.item())
```

**Structure:** Preprocess → Inference → Postprocess. Each stage should be profiled independently to find bottlenecks.

---

### 6.5 Running with Uvicorn

```bash
# Development (with auto-reload)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Production (multiple workers)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

| Parameter        | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `--host 0.0.0.0` | Listen on all interfaces (needed inside containers)  |
| `--port 8000`    | HTTP port                                            |
| `--workers 4`    | Number of worker processes (for CPU-bound inference) |
| `--reload`       | Auto-restart on code change (development only)       |

For GPU inference, typically use 1 worker per GPU (the GPU handles parallelism internally).

---

## 7. Containerisation with Docker

### 7.1 Why Containers?

| Problem               | Docker solution                              |
| --------------------- | -------------------------------------------- |
| "Works on my machine" | Identical environment everywhere             |
| Dependency conflicts  | Isolated filesystem and libraries            |
| Deployment complexity | Single artifact (image) to ship              |
| Scaling               | Orchestrate with Docker Compose / Kubernetes |

A container packages the OS libraries, Python, dependencies, model file, and application code into a single, reproducible image.

---

### 7.2 Dockerfile Anatomy

```dockerfile
# 1. Base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code and model
COPY app.py .
COPY model_weights.pt .

# 5. Expose port
EXPOSE 8000

# 6. Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Layer caching:** Docker caches each instruction. By copying `requirements.txt` before the source code, dependency installation is cached unless requirements change — this makes rebuilds fast.

---

### 7.3 Building and Running

```bash
# Build the image
docker build -t ml-api:latest .

# Run the container
docker run -p 8000:8000 ml-api:latest

# Test
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie is excellent!"}'
```

---

### 7.4 Best Practices

| Practice                                 | Reason                                                  |
| ---------------------------------------- | ------------------------------------------------------- |
| Use `python:3.11-slim` (not full)        | Reduces image from ~900 MB to ~150 MB                   |
| Pin dependency versions                  | Reproducibility                                         |
| Use `.dockerignore`                      | Exclude `__pycache__`, `.git`, notebooks, training data |
| Don't include training data in the image | Only the model checkpoint and serving code              |
| Use multi-stage builds for compiled deps | Smaller final image                                     |
| Run as non-root user                     | Security                                                |
| Set `PYTHONUNBUFFERED=1`                 | Ensure print/log output appears immediately             |

---

## 8. Health Checks and Monitoring

### 8.1 Health Endpoints

Two standard endpoints:

```python
@app.get('/health', response_model=HealthResponse)
def health():
    return HealthResponse(
        status='ok',
        model_loaded=hasattr(app.state, 'model')
    )

@app.get('/ready')
def ready():
    if not hasattr(app.state, 'model'):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {'status': 'ready'}
```

| Endpoint             | Purpose                   | Used by                                       |
| -------------------- | ------------------------- | --------------------------------------------- |
| `/health` (liveness) | "Is the process running?" | Container orchestrator (restart if unhealthy) |
| `/ready` (readiness) | "Can it handle requests?" | Load balancer (route traffic only when ready) |

---

### 8.2 Request Latency Logging

Middleware that measures every request:

```python
import time
from fastapi import Request

@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    print(f"{request.method} {request.url.path} — {latency*1000:.1f}ms")
    return response
```

This logs every request's method, path, and latency in milliseconds. In production, send these to a structured log system (JSON logs → ELK stack, CloudWatch, etc.).

---

### 8.3 Prometheus Metrics

For more sophisticated monitoring, expose metrics in Prometheus format:

```python
from prometheus_client import Counter, Histogram, make_asgi_app

REQUEST_COUNT = Counter('predict_requests_total', 'Total prediction requests')
REQUEST_LATENCY = Histogram('predict_latency_seconds', 'Prediction latency')

@app.post('/predict')
def predict(request: PredictionRequest):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        # ... inference ...
        return result

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount('/metrics', metrics_app)
```

Prometheus scrapes `/metrics` at regular intervals. Grafana visualises the time-series data.

| Metric type   | Example                   | Use                                  |
| ------------- | ------------------------- | ------------------------------------ |
| **Counter**   | `predict_requests_total`  | Total request count (monotonic)      |
| **Histogram** | `predict_latency_seconds` | Latency distribution (p50, p95, p99) |
| **Gauge**     | `model_loaded`            | Current state (0 or 1)               |

---

### 8.4 What to Monitor

| Category       | Metrics                                          | Why                              |
| -------------- | ------------------------------------------------ | -------------------------------- |
| **Latency**    | p50, p95, p99 response time                      | Detect slowdowns, SLA compliance |
| **Throughput** | Requests per second                              | Capacity planning                |
| **Errors**     | 4xx/5xx rate                                     | Detect bugs, input issues        |
| **Model**      | Prediction distribution, confidence distribution | Detect data drift                |
| **System**     | CPU/GPU utilisation, memory, disk                | Detect resource exhaustion       |

**The golden signals** (from Google SRE):
1. **Latency** — how long requests take.
2. **Traffic** — how many requests per second.
3. **Errors** — how many requests fail.
4. **Saturation** — how "full" the system is.

---

## 9. Model Serving Frameworks

Beyond hand-rolled FastAPI:

| Framework                   | Key features                                                          | Best for                     |
| --------------------------- | --------------------------------------------------------------------- | ---------------------------- |
| **FastAPI + Uvicorn**       | Simple, flexible, Python-native                                       | Small services, prototypes   |
| **TorchServe**              | PyTorch-native, batching, model versioning                            | PyTorch models in production |
| **Triton Inference Server** | Multi-framework (PyTorch, TF, ONNX), dynamic batching, GPU scheduling | High-throughput GPU serving  |
| **vLLM**                    | Continuous batching, PagedAttention, KV-cache                         | LLM serving                  |
| **BentoML**                 | Model packaging, versioning, deployment                               | MLOps-focused teams          |
| **TensorFlow Serving**      | gRPC/REST, model versioning                                           | TensorFlow models            |

For this course, FastAPI is sufficient. Production systems at scale typically use Triton or TorchServe.

---

## 10. Data and Model Drift

A model's performance degrades over time as the world changes.

**Data drift** ($P(x)$ changes): the distribution of inputs shifts from what the model was trained on. Example: a sentiment model trained on product reviews starts receiving social media text.

**Concept drift** ($P(y|x)$ changes): the relationship between inputs and labels changes. Example: "sick" becomes slang for "cool" — the mapping from word to sentiment shifts.

**Detection strategies:**

| Method                           | Detects    | How                                                                                            |
| -------------------------------- | ---------- | ---------------------------------------------------------------------------------------------- |
| Monitor prediction distribution  | Data drift | Plot confidence histograms over time; increasing low-confidence predictions signals OOD inputs |
| Statistical tests (KS test, PSI) | Data drift | Compare feature distributions between training data and recent production data                 |
| Labelled evaluation set          | Both       | Periodically collect ground-truth labels and compute metrics                                   |
| Confidence calibration ([Week 08](../../03_probability/week08_uncertainty/theory.md#9-calibration)) | Both       | If calibrated probabilities become unreliable, the model is degrading                          |

**Response:** retrain on fresh data (closing the lifecycle loop).

---

## 11. CI/CD for ML

Continuous Integration / Continuous Deployment adapted for ML:

```
Code change
    │
    ▼
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│   Lint    │───►│   Test    │───►│   Build   │───►│  Deploy   │
│ & Format  │    │  & Eval   │    │   Image   │    │  (Staged) │
└───────────┘    └───────────┘    └───────────┘    └───────────┘
```

| Stage           | What it does                                   | Tools                          |
| --------------- | ---------------------------------------------- | ------------------------------ |
| Lint & Format   | Code quality                                   | `ruff`, `black`, `mypy`        |
| Test & Eval     | Unit tests + model evaluation on held-out data | `pytest`, evaluation script    |
| Build Image     | Docker build + push to registry                | `docker build`, GitHub Actions |
| Deploy (Staged) | Canary or blue-green deployment                | Kubernetes, cloud platforms    |

**ML-specific CI additions:**
- **Model evaluation gate:** deploy only if metrics exceed a threshold.
- **Data validation:** check that new training data matches expected schema and distributions.
- **Model registry:** track model versions, metadata, and lineage (MLflow, W&B).

---

## 12. Security and Validation

| Concern                        | Mitigation                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| **Malicious input**            | Validate with Pydantic schemas; limit input size (`max_length`)                      |
| **Denial of service**          | Rate limiting; request size limits; timeout on inference                             |
| **Model theft**                | Don't expose raw logits (return only labels + confidence); authentication (API keys) |
| **Pickle deserialization**     | Use `weights_only=True` or SafeTensors; never load untrusted pickles                 |
| **Dependency vulnerabilities** | Pin versions; audit with `pip-audit` or `safety`                                     |
| **Container security**         | Run as non-root; use minimal base images; scan with `trivy`                          |

Input validation example:

```python
class PredictionRequest(BaseModel):
    text: str = Field(..., max_length=1000, description="Input text")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text must not be empty')
        return v
```

---

## 13. Connections to the Rest of the Course

| Week                               | Connection                                                                                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Week 01](../../02_fundamentals/week01_optimization/theory.md#4-gradient-descent)–[Week 02](../../02_fundamentals/week02_advanced_optimizers/theory.md#81-adamw-decoupled-weight-decay) (Optimisation)**      | The model you deploy was optimised with gradient descent; the loss landscape determines the quality of the deployed model                           |
| **[Week 06](../../02_fundamentals/week06_regularization/theory.md#3-ridge-regression-l2-regularisation) (Regularisation)**       | A well-regularised model generalises better in production; overfitting manifests as poor predictions on real-world inputs                           |
| **[Week 08](../../03_probability/week08_uncertainty/theory.md#9-calibration) (Uncertainty)**          | Calibrated confidence scores are essential for production systems; users need to know when to trust predictions                                     |
| **[Week 12](../../04_neural_networks/week12_training_pathologies/theory.md#3-vanishing-gradients) (Training Pathologies)** | A model with training pathologies (vanishing gradients, dead neurons) produces poor predictions; diagnosis happens before deployment                |
| **[Week 14](../../05_deep_learning/week14_training_at_scale/theory.md#5-robust-checkpointing) (Training at Scale)**    | Checkpointing gives you the model file; mixed precision training produces FP16-compatible weights ready for quantised serving                       |
| **[Week 16](../../05_deep_learning/week16_regularization_dl/theory.md#3-dropout) (Regularisation DL)**    | Dropout must be disabled at inference (`model.eval()`); ensembles can improve production predictions                                                |
| **[Week 18](../../06_sequence_models/week18_transformers/theory.md#2-the-transformer-at-a-glance) (Transformers)**         | The Transformer is likely your deployed model; knowledge of its architecture informs optimisation choices (which layers to quantise, what to prune) |
| **[Week 19](../../07_transfer_learning/week19_finetuning/theory.md#3-adaptation-strategies) (Fine-Tuning)**          | The fine-tuned model is your deployment candidate; LoRA enables serving multiple tasks from one backbone                                            |
| **Capstone**                       | This week's deliverables (API, Dockerfile, deploy.md) are the capstone deployment artifacts                                                         |

---

## 14. Notebook Reference Guide

| Cell                     | Section              | What it demonstrates                                                                                 | Theory reference |
| ------------------------ | -------------------- | ---------------------------------------------------------------------------------------------------- | ---------------- |
| 1 (Setup)                | FastAPI app skeleton | Imports, `FastAPI()`, `InputPayload` Pydantic model, cache utilities, `/health` and `/predict` stubs | Sections 6.1–6.4 |
| Ex. 1 (Minimal API)      | Serving              | Run with `uvicorn`, send test JSON, measure latency                                                  | Sections 6.4–6.5 |
| Ex. 2 (Containerise)     | Docker               | Write `Dockerfile`, build image, run container                                                       | Section 7        |
| Ex. 3 (Health & Metrics) | Monitoring           | `/health` endpoint, latency middleware, optional Prometheus                                          | Section 8        |
| Deliverables             | Checklist            | API in `capstone/src/`, `Dockerfile`, `deploy.md`, health endpoint                                   | —                |

**Suggested modifications:**

| Modification                                                                                      | What it reveals                                                              |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Add dynamic quantisation (`quantize_dynamic`) and compare inference time and model size           | INT8 models are ~4× smaller and often 1.5–2× faster on CPU                   |
| Export the model to ONNX and serve with ONNX Runtime instead of PyTorch                           | Typically 1.5–3× speedup from graph optimisations                            |
| Implement dynamic batching: buffer requests for 50ms, then process as a batch                     | Throughput increases significantly at the cost of added latency              |
| Add a `/metrics` endpoint with Prometheus and visualise in Grafana                                | Full observability stack; see latency percentiles over time                  |
| Simulate data drift by sending inputs from a different domain and monitor confidence distribution | Confidence drops, prediction distribution shifts — triggers retraining alert |
| Implement a canary deployment: route 10% of traffic to the new model version                      | Safe rollout strategy; compare metrics between old and new models            |
| Load-test with `wrk` or `locust` at increasing concurrency levels                                 | Find the throughput ceiling and latency degradation curve                    |
| Add input validation that rejects sequences > 512 tokens with a 400 error                         | Security best practice: prevents OOM from adversarial inputs                 |

---

## 15. Symbol Reference

| Symbol                 | Name                       | Meaning                                                                     |
| ---------------------- | -------------------------- | --------------------------------------------------------------------------- |
| $t_\text{preprocess}$  | Preprocessing time         | Time to convert raw input to model input (tokenisation, normalisation)      |
| $t_\text{inference}$   | Inference time             | Time for the forward pass through the model                                 |
| $t_\text{postprocess}$ | Postprocessing time        | Time to convert model output to the response format                         |
| $N_\text{params}$      | Parameter count            | Number of learnable weights in the model                                    |
| $T$                    | Temperature (distillation) | Softmax temperature for knowledge distillation; higher = softer predictions |
| $\alpha$               | Distillation weight        | Balance between hard label loss and soft distillation loss                  |
| $\tau$                 | Pruning threshold          | Weights below $\tau$ are set to zero                                        |
| RPS / QPS              | Requests per second        | Throughput metric                                                           |
| p50, p95, p99          | Latency percentiles        | Median, 95th, 99th percentile response times                                |
| PSI                    | Population Stability Index | Measures distribution shift between reference and production data           |
| KS test                | Kolmogorov–Smirnov test    | Non-parametric test for whether two distributions differ                    |

---

## 16. References

1. Google SRE Book, Chapter 6: "Monitoring Distributed Systems." https://sre.google/sre-book/monitoring-distributed-systems/ — The four golden signals: latency, traffic, errors, saturation.
2. Sculley, D., Holt, G., Golovin, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*. — ML systems are more than the model: data pipelines, monitoring, configuration.
3. Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2022). "Challenges in Deploying Machine Learning: A Survey of Case Studies." *ACM Computing Surveys*. — Practical deployment challenges.
4. FastAPI Documentation. https://fastapi.tiangolo.com/ — Official docs for the framework used in the notebook.
5. Docker Documentation. https://docs.docker.com/ — Container fundamentals.
6. Jacob, B., Kligys, S., Chen, B., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*. — Quantisation-aware training.
7. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *NeurIPS Workshop*. — Knowledge distillation; temperature scaling; dark knowledge.
8. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). "DistilBERT, a Distilled Version of BERT." *NeurIPS Workshop*. — Practical distillation applied to BERT.
9. ONNX Runtime Documentation. https://onnxruntime.ai/ — Optimised inference engine.
10. Kwon, W., Li, Z., Zhuang, S., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP*. — vLLM; continuous batching for LLM serving.
11. Prometheus Documentation. https://prometheus.io/docs/ — Time-series monitoring and alerting.
12. Huyen, C. (2022). *Designing Machine Learning Systems.* O'Reilly. — End-to-end ML system design, covering data pipelines through deployment and monitoring.
