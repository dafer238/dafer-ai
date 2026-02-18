Week 10 — Efficient Training (Training at Scale)

## Prerequisites

- **Week 09** — PyTorch basics (`nn.Module`, training loop, optimizer). Week 10 wraps that loop in production tooling; you need the bare loop working first.
- **Week 01–02** — learning rate intuition. Schedulers only make sense once you understand why the learning rate matters.

## What this week delivers

A training loop that works on a toy example is not the same as one that works reliably on a real dataset. Week 10 provides the engineering layer: efficient data loading, learning rate schedules, and robust checkpointing. These are the habits that separate research scripts from reproducible, resumable experiments.

Overview
Focus on building efficient training pipelines: batching, data I/O, schedulers, checkpointing, and simple distributed ideas.

Study
- Batching strategies, shuffling, prefetching, and memory footprint
- Learning rate schedulers and checkpointing strategies

Practical libraries & tools
- PyTorch DataLoader, `torch.utils.data.DataLoader`, `torch.distributed` (overview)
- Datasets: small local datasets for profiling; torchvision utilities

Exercises
1) Implement efficient DataLoader with custom Dataset and prefetching

2) Measure throughput
   - Time data loading vs GPU compute; identify bottlenecks.

3) Checkpointing and resume
   - Implement robust checkpoint save/load with metadata.

Code hint
  from torch.utils.data import DataLoader
  loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

Reading
- Papers/guides on data pipelines, profiling tips, and distributed training primers.

Deliverable
- Notebook/script measuring throughput and demonstrating an efficient, resumable training loop.

## What comes next

- **Week 11** builds CNNs using the DataLoader and training-loop patterns established here.
- **Week 12** adds regularization (Dropout, BatchNorm, augmentation) as modifications to the same pipeline.
