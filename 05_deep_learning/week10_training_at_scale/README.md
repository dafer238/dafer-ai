Week 10 — Efficient Training (Training at Scale)

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
