Week 16 — Deployment & Capstone

## Prerequisites

- **All previous weeks** — you need a trained, evaluated model to deploy.
- **Week 04** (evaluation discipline) and **Week 06** (uncertainty) — a deployed system without proper evaluation and uncertainty reporting is dangerous, not useful.

## What this week delivers

The full ML lifecycle: data → model → evaluate → deploy → monitor. You have built every component in isolation; this week you connect them. The capstone is the proof.

Overview
Finalize an end-to-end demo: package a trained model, expose a simple API for inference, and show monitoring and basic lifecycle management.

Study
- Inference constraints: latency, memory, and throughput
- Model serving options: Flask, FastAPI, TorchServe, and lightweight containers
- Monitoring basics: metrics, logs, and health checks

Practical libraries & tools
- FastAPI or Flask for simple APIs
- Docker for containerization
- Prometheus/Grafana (optional) for metrics
- `uvicorn` for ASGI serving

Exercises
1) Minimal API
   - Build a small FastAPI app that loads a saved model and returns predictions for JSON inputs.

2) Containerize
   - Write a `Dockerfile` for the API and run locally.

3) Health & metrics
   - Add a `/health` endpoint and simple request latency metric logging.

4) Basic CI/Deployment
   - Provide instructions to build the Docker image and run it locally; optionally push to a registry.

Code hints
- FastAPI example

  from fastapi import FastAPI
  import torch

  app = FastAPI()
  model = torch.load('model.pt')

  @app.post('/predict')
  def predict(payload: dict):
      # convert payload to tensor, run model, return results
      pass

Docker hint
- Use a slim Python base image, copy requirements, install, and expose the port.

Deliverable
- `capstone/src/` minimal API (or script), `Dockerfile`, and short `deploy.md` with instructions to run locally.

Notes
- Keep the demo small and reproducible; prioritize clarity over production readiness.
