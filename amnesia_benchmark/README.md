# The Amnesia Benchmark & FWP Proof of Concept

This directory contains the initial proof of concept for **Ogma**, demonstrating the fundamental flaw in standard neural networks and proving the viability of the **Fast Weight Programmer (FWP)** architecture.

## The Core Problem: "Amnesia" in Standard AI
Standard neural networks (like RNNs, LSTMs, and basic Transformers) rely on **slow weights** updated via backpropagation. If you try to teach them rapidly changing rules—such as `A=5` in one sequence, and `A=1` in the next—they fail. They suffer from "amnesia" because they attempt to permanently etch temporary rules into their static weights.

### 1. The Dataset (`dataset.py`)
To prove this, we built a synthetic **Key-Value Memory Task**. 
It generates random, non-repeating rules on the fly:
* **Input:** `A=5, Y=2, C=1, E=8, ?E`
* **Target:** `8`
The rules change every single time the script runs. The network must hold the sequence in a temporary buffer, look at the query, and retrieve the correct answer without relying on long-term memory.

### 2. The Failing Baseline (`baseline.py`)
We implemented a standard PyTorch LSTM and trained it on the Amnesia dataset. 
* **Result:** The LSTM fails. It rapidly plateaus at around 30-40% accuracy. Because the rules contradict themselves across sequences, the LSTM's static weights are completely incapable of dynamically binding the keys to the values.

### 3. The Solution: Ogma's Fast Weight Programmer (`ogma.py`)
Instead of relying on slow weights, the Ogma cell implements a **Delta-Rule Update**:
1. It reads the input and uses a "Slow Net" to provide local context.
2. It generates a **Key** and a **Value** vector.
3. It uses the mathematical **Outer Product** ($V \otimes K^T$) to create a dynamic weight matrix ($\Delta W$).
4. It instantly adds this to a blank "whiteboard" (its Fast Weight memory).
5. When queried, it multiplies the Fast Weight matrix by a **Query** vector to instantly retrieve the correct value.

* **Result:** The FWP achieves **~96-100% accuracy** in just a few epochs, absolutely crushing the LSTM baseline. It successfully invents a differentiable dictionary on the fly using pure linear algebra.

---

## How to Run the Tests

Ensure you have a Python environment with PyTorch installed.

1. **Test the Dataset:**
   ```bash
   python dataset.py
   ```
2. **Run the Failing Baseline:**
   ```bash
   python baseline.py
   ```
3. **Run the Ogma Network:**
   ```bash
   python ogma.py
   ```
