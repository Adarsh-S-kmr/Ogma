# Ogma Phase 2: Character-Level Language Modeling

This phase demonstrates the Ogma architecture scaling up from synthetic key-value retrieval to a real-world continuous data stream: autoregressive language modeling.

## The Goal
The purpose of this phase is to prove that the **Delta-Rule Update** and the Fast Weight memory matrix can naturally act like the KV-cache of a Transformer, building contextual representations over time without requiring the massive $O(N^2)$ attention mechanism.

## Files
1. `dataset.py`: Automatically downloads the `tiny-shakespeare` dataset (1MB) and handles tokenization and batching into sequences.
2. `transformer_baseline.py`: A standard PyTorch Transformer Decoder (GPT-style) to act as a baseline comparison for parameter count and loss convergence.
3. `ogma_lm.py`: The Ogma Fast Weight Programmer updated for autoregressive generation. At each step $t$, the matrix $W$ is updated, and the new memory is queried to predict the $t+1$ character.

## How to run
You can run these scripts using your virtual environment:
```bash
..\amnesia_benchmark\venv\Scripts\python.exe dataset.py
..\amnesia_benchmark\venv\Scripts\python.exe transformer_baseline.py
..\amnesia_benchmark\venv\Scripts\python.exe ogma_lm.py
```
