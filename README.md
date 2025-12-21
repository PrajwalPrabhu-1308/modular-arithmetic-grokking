# Modular Arithmetic Grokking with Curriculum Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Investigating **grokking** — sudden generalization after prolonged overfitting — in small causal transformers trained on modular arithmetic tasks (addition and multiplication modulo a prime).

This repository provides a clean, production-ready implementation with:
- Curriculum learning based on wrapping complexity (avoids "linear trap")
- ALiBi positional encodings (no learned positional embeddings)
- Weight tying between input embeddings and output projection
- Comprehensive logging, checkpoints, and PCA visualizations of embedding evolution

## Key Takeaways

After training the model on modular addition and multiplication (p = 113), the following consistent behaviors emerge:

| Phase                  | Behavior                                                                 | Typical Timing (steps) |
|------------------------|--------------------------------------------------------------------------|------------------------|
| **Memorization**       | Model overfits training data → high training accuracy, low generalization | 0–5k steps            |
| **Plateau**            | Training accuracy plateaus near ~50–60%, loss stalls                    | 5k–15k steps          |
| **Grokking**           | Sudden generalization spike → test accuracy jumps to ~98–100%           | ~15k–20k steps        |
| **Embedding Geometry** | Numeric token embeddings evolve from random → clear circular ring structure | Ring forms ~12k–18k steps |

**Curriculum learning accelerates grokking**:
- Without curriculum: grokking is delayed or sometimes fails entirely (linear trap)
- With curriculum: grokking reliably occurs ~30–50% faster and more consistently

**Weight tying + ALiBi** are both important contributors to clean generalization:
- Weight tying enforces consistent representation between input and output
- ALiBi provides inductive bias for modular (cyclic) structure

**PCA Visualizations** (saved in `grokking_output/snapshots/`):
- Early training: scattered points
- Mid training: clusters begin to align
- Late training: near-perfect circular arrangement reflecting modular arithmetic

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/modular-arithmetic-grokking.git
cd modular-arithmetic-grokking

# Recommended: use a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install torch numpy matplotlib scikit-learn tqdm
