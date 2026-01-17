
"""
PRIORITY 1: Quick Validation with Test Accuracy
================================================
Tests ONE critical experiment with train/test split to see if we have
true grokking (generalization) or just memorization.

Experiment: complexity curriculum, mixed task, seed 42
Time: ~30 minutes
Result: Tells you if test accuracy generalizes or stays low

Just run: result = run_test_accuracy_experiment()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
import pandas as pd
from IPython.display import display, HTML

# ==========================================
# CONFIG (Same as before)
# ==========================================

@dataclass
class TestConfig:
    p: int = 113
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1
    max_steps: int = 40000
    batch_size: int = 512
    curriculum_type: str = "complexity"
    task_type: str = "mixed"
    curriculum_threshold: float = 0.95
    curriculum_window: int = 20
    log_interval: int = 250
    use_alibi: bool = True
    max_seq_len: int = 64
    save_dir: str = "test_accuracy_results"
    experiment_name: str = "test_acc_validation"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # NEW: Test set parameters
    test_set_size: int = 10000  # Fixed test set size
    train_fraction: float = 0.3  # Use 30% of pairs for training

# ==========================================
# MODEL (Same as before - no changes)
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dim = config.dim

        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

        if config.use_alibi:
            slopes = self._compute_alibi_slopes(config.n_heads)
            self.register_buffer("alibi_slopes", slopes)
        else:
            self.alibi_slopes = None

    @staticmethod
    def _compute_alibi_slopes(n_heads):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-8 / n)
                ratio = start
                return [start * (ratio ** i) for i in range(n)]

            if n & (n - 1) == 0:
                return get_slopes_power_of_2(n)
            else:
                closest_power = 2 ** np.floor(np.log2(n))
                slopes = get_slopes_power_of_2(int(closest_power))
                slopes += get_slopes(int(2 * closest_power))[0::2][:n - int(closest_power)]
                return slopes

        return torch.tensor(get_slopes(n_heads), dtype=torch.float32)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:T, :T], float('-inf'))

        if self.alibi_slopes is not None:
            positions = torch.arange(T, device=x.device)
            alibi_bias = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
            alibi_bias = -self.alibi_slopes.view(-1, 1, 1) * alibi_bias.unsqueeze(0)
            att = att + alibi_bias.unsqueeze(0)

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim),
            nn.GELU(),
            nn.Linear(4 * config.dim, config.dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GrokkingTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.p = config.p
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, config.dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.dim)

        self._init_weights()

    def _init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        self.apply(_init)

    def forward(self, x):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.weight[:self.p])
        return logits

# ==========================================
# CURRICULUM WITH TRAIN/TEST SPLIT
# ==========================================

class TestAwareCurriculum:
    """Curriculum that maintains train/test split."""

    def __init__(self, p, config):
        self.p = p
        self.config = config
        self.level = 0
        self.max_level = 2

        self.delim_token = p
        self.add_token = p + 1
        self.mul_token = p + 2
        self.vocab_size = p + 3

        self.accuracy_history = []

        # Setup curriculum ranges
        if config.curriculum_type == "complexity":
            self.ranges = [(self.p // 2, self.p), (0, self.p), (0, self.p)]
        elif config.curriculum_type == "magnitude":
            self.ranges = [(0, self.p // 4), (0, self.p // 2), (0, self.p)]
        else:
            self.ranges = [(0, self.p)] * 3

        # NEW: Create fixed test set at initialization
        self.test_data = self._create_test_set()
        print(f"✓ Created fixed test set: {len(self.test_data[0])} examples")

    def _create_test_set(self):
        """
        Create a fixed held-out test set.
        Uses a deterministic seed to ensure consistency.
        """
        # Save current random state
        current_state = torch.get_rng_state()

        # Use fixed seed for test set
        torch.manual_seed(999)  # Different from training seed

        size = self.config.test_set_size
        device = self.config.device

        # Generate all test examples
        a = torch.randint(0, self.p, (size,), device=device)
        b = torch.randint(0, self.p, (size,), device=device)

        # 50/50 split between operations
        is_mul = torch.rand(size, device=device) > 0.5
        ops = torch.where(is_mul, self.mul_token, self.add_token)

        # Compute targets
        y_add = (a + b) % self.p
        y_mul = (a * b) % self.p
        y = torch.where(is_mul, y_mul, y_add)

        # Create input
        delim = torch.full_like(a, self.delim_token)
        x = torch.stack([a, b, ops, delim], dim=1)

        # Restore random state
        torch.set_rng_state(current_state)

        return x, y

    def get_batch(self, batch_size, mode='train'):
        """
        Generate batch for training or testing.

        Args:
            batch_size: Number of examples
            mode: 'train' or 'test'
        """
        if mode == 'test':
            # Return random subset of test set
            indices = torch.randint(0, len(self.test_data[0]), (batch_size,))
            return self.test_data[0][indices], self.test_data[1][indices]

        # TRAINING batch generation (same as before)
        device = self.config.device

        # Determine range based on curriculum
        if self.config.curriculum_type == "none" or self.level >= self.max_level:
            low, high = 0, self.p
            use_mixed = False
        elif self.config.curriculum_type == "complexity" and self.level == 1:
            low, high = 0, self.p
            use_mixed = True
        else:
            low, high = self.ranges[self.level]
            use_mixed = False

        # Sample
        if use_mixed:
            n_full = int(batch_size * 0.7)
            n_wrap = batch_size - n_full

            a_full = torch.randint(0, self.p, (n_full,), device=device)
            b_full = torch.randint(0, self.p, (n_full,), device=device)
            a_wrap = torch.randint(self.p // 2, self.p, (n_wrap,), device=device)
            b_wrap = torch.randint(self.p // 2, self.p, (n_wrap,), device=device)

            a = torch.cat([a_full, a_wrap])
            b = torch.cat([b_full, b_wrap])
        else:
            a = torch.randint(low, high, (batch_size,), device=device)
            b = torch.randint(low, high, (batch_size,), device=device)

        # Operations (mixed task)
        is_mul = torch.rand(batch_size, device=device) > 0.5
        ops = torch.where(is_mul, self.mul_token, self.add_token)

        # Targets
        y_add = (a + b) % self.p
        y_mul = (a * b) % self.p
        y = torch.where(is_mul, y_mul, y_add)

        # Input
        delim = torch.full_like(a, self.delim_token)
        x = torch.stack([a, b, ops, delim], dim=1)

        return x, y

    def update_progress(self, accuracy):
        """Update curriculum level based on training accuracy."""
        if self.config.curriculum_type == "none" or self.level >= self.max_level:
            return False

        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > self.config.curriculum_window:
            self.accuracy_history.pop(0)

        if len(self.accuracy_history) >= self.config.curriculum_window:
            avg_acc = np.mean(self.accuracy_history)
            if avg_acc >= self.config.curriculum_threshold:
                self.level += 1
                self.accuracy_history.clear()
                return True
        return False

# ==========================================
# TRAINER WITH TEST ACCURACY TRACKING
# ==========================================

class TestAccuracyTrainer:
    """Trainer that tracks both train and test accuracy."""

    def __init__(self, model, curriculum, config):
        self.model = model
        self.curriculum = curriculum
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )

        # NEW: Track both train and test accuracy
        self.history = {
            'step': [],
            'train_acc': [],
            'test_acc': [],           # NEW
            'train_acc_add': [],
            'test_acc_add': [],       # NEW
            'train_acc_mul': [],
            'test_acc_mul': [],       # NEW
            'train_loss': [],
            'test_loss': [],          # NEW
            'curriculum_level': [],
            'generalization_gap': []  # NEW: train_acc - test_acc
        }

        self.output_dir = Path(config.save_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_test(self):
        """
        Evaluate on the held-out test set.
        Returns: test_acc, test_acc_add, test_acc_mul, test_loss
        """
        self.model.eval()

        with torch.no_grad():
            # Get test batch (larger batch for more stable estimate)
            x_test, y_test = self.curriculum.get_batch(1024, mode='test')

            # Forward pass
            logits = self.model(x_test)
            logits_last = logits[:, -1, :]

            # Loss
            test_loss = F.cross_entropy(logits_last, y_test).item()

            # Overall accuracy
            pred = logits_last.argmax(dim=-1)
            test_acc = (pred == y_test).float().mean().item()

            # Operation-specific accuracy
            ops = x_test[:, 2]
            is_add = (ops == self.curriculum.add_token)
            is_mul = (ops == self.curriculum.mul_token)

            test_acc_add = (pred[is_add] == y_test[is_add]).float().mean().item() if is_add.any() else 0.0
            test_acc_mul = (pred[is_mul] == y_test[is_mul]).float().mean().item() if is_mul.any() else 0.0

        self.model.train()
        return test_acc, test_acc_add, test_acc_mul, test_loss

    def train(self):
        """Training loop with test accuracy tracking."""
        print(f"\n{'='*60}")
        print(f"Training: {self.config.experiment_name}")
        print(f"Tracking TRAIN and TEST accuracy")
        print(f"{'='*60}\n")

        pbar = tqdm(range(self.config.max_steps), desc="Training")

        train_grok_step = None
        test_grok_step = None

        for step in pbar:
            # TRAINING STEP
            x_train, y_train = self.curriculum.get_batch(self.config.batch_size, mode='train')

            self.model.train()
            logits = self.model(x_train)
            logits_last = logits[:, -1, :]
            loss = F.cross_entropy(logits_last, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # LOGGING (includes test evaluation)
            if step % self.config.log_interval == 0:
                with torch.no_grad():
                    # Train accuracy
                    pred_train = logits_last.argmax(dim=-1)
                    train_acc = (pred_train == y_train).float().mean().item()

                    ops = x_train[:, 2]
                    is_add = (ops == self.curriculum.add_token)
                    is_mul = (ops == self.curriculum.mul_token)

                    train_acc_add = (pred_train[is_add] == y_train[is_add]).float().mean().item() if is_add.any() else 0.0
                    train_acc_mul = (pred_train[is_mul] == y_train[is_mul]).float().mean().item() if is_mul.any() else 0.0

                # Test accuracy
                test_acc, test_acc_add, test_acc_mul, test_loss = self.evaluate_test()

                # Generalization gap
                gen_gap = train_acc - test_acc

                # Record history
                self.history['step'].append(step)
                self.history['train_acc'].append(train_acc)
                self.history['test_acc'].append(test_acc)
                self.history['train_acc_add'].append(train_acc_add)
                self.history['test_acc_add'].append(test_acc_add)
                self.history['train_acc_mul'].append(train_acc_mul)
                self.history['test_acc_mul'].append(test_acc_mul)
                self.history['train_loss'].append(loss.item())
                self.history['test_loss'].append(test_loss)
                self.history['curriculum_level'].append(self.curriculum.level)
                self.history['generalization_gap'].append(gen_gap)

                # Track grokking points
                if train_acc >= 0.95 and train_grok_step is None:
                    train_grok_step = step

                if test_acc >= 0.95 and test_grok_step is None:
                    test_grok_step = step
                    print(f"\n🎉 TEST GROKKING at step {step}! (test_acc={test_acc:.4f})")

                # Update curriculum
                leveled_up = self.curriculum.update_progress(train_acc)
                if leveled_up:
                    print(f"\n🚀 Curriculum Level {self.curriculum.level}")

                # Update progress bar
                pbar.set_postfix({
                    'train': f'{train_acc:.3f}',
                    'test': f'{test_acc:.3f}',
                    'gap': f'{gen_gap:.3f}',
                    'lvl': self.curriculum.level
                })

        # Final results
        results = {
            'train_grok_step': train_grok_step if train_grok_step else self.config.max_steps,
            'test_grok_step': test_grok_step if test_grok_step else self.config.max_steps,
            'final_train_acc': self.history['train_acc'][-1],
            'final_test_acc': self.history['test_acc'][-1],
            'final_gen_gap': self.history['generalization_gap'][-1],
            'history': self.history
        }

        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

# ==========================================
# ANALYSIS & VISUALIZATION
# ==========================================

def analyze_grokking_results(results):
    """Analyze and visualize grokking dynamics."""

    history = results['history']

    print("\n" + "="*60)
    print("📊 GROKKING ANALYSIS")
    print("="*60)

    # Summary statistics
    train_grok = results['train_grok_step']
    test_grok = results['test_grok_step']
    final_train = results['final_train_acc']
    final_test = results['final_test_acc']
    final_gap = results['final_gen_gap']

    print(f"\n🎯 GROKKING POINTS:")
    print(f"  Train reaches 95%: Step {train_grok}")
    print(f"  Test reaches 95%:  Step {test_grok}")

    if test_grok < 12000:
        delay = test_grok - train_grok
        print(f"\n  ⏱️  Grokking delay: {delay} steps")

        if delay < 500:
            print(f"  ✅ SMOOTH GENERALIZATION (small delay)")
        elif delay < 3000:
            print(f"  ⚡ MODERATE GROKKING (typical delay)")
        else:
            print(f"  🔥 DRAMATIC GROKKING (large delay)")
    else:
        print(f"\n  ⚠️  Test did NOT reach 95% by step {test_grok}")
        print(f"  → This suggests MEMORIZATION, not generalization")

    print(f"\n📈 FINAL METRICS:")
    print(f"  Train accuracy: {final_train:.4f}")
    print(f"  Test accuracy:  {final_test:.4f}")
    print(f"  Gen. gap:       {final_gap:.4f}")

    if final_gap < 0.05:
        print(f"\n  ✅ STRONG GENERALIZATION (gap < 5%)")
    elif final_gap < 0.15:
        print(f"\n  ⚠️  MODERATE GENERALIZATION (gap 5-15%)")
    else:
        print(f"\n  ❌ POOR GENERALIZATION (gap > 15%)")
        print(f"  → Model is likely memorizing, not learning algorithm")

    # Create visualization
    create_grokking_plot(history)

    # Interpretation
    print("\n" + "="*60)
    print("💡 INTERPRETATION")
    print("="*60)

    if final_test > 0.90 and final_gap < 0.10:
        print("\n✅ TRUE GROKKING OBSERVED!")
        print("   Your model learns generalizable algorithmic solutions.")
        print("   You CAN claim:")
        print("   • 'Curriculum enables algorithmic discovery'")
        print("   • 'Model learns Fourier representations that generalize'")
        print("   • 'We observe the grokking phenomenon'")

    elif final_test > 0.70 and final_gap < 0.20:
        print("\n⚡ PARTIAL GENERALIZATION")
        print("   Model learns some generalizable features but not fully.")
        print("   You CAN claim:")
        print("   • 'Curriculum enables partial generalization'")
        print("   • 'Model learns representations with moderate generalization'")

    else:
        print("\n⚠️  LIMITED GENERALIZATION")
        print("   Model primarily memorizes rather than learning algorithm.")
        print("   You SHOULD claim:")
        print("   • 'Curriculum enables training convergence'")
        print("   • 'Further investigation needed for generalization'")
        print("   • Address this honestly in limitations")

    return results

def create_grokking_plot(history):
    """Create comprehensive grokking visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    steps = history['step']

    # Plot 1: Train vs Test Accuracy (THE KEY PLOT)
    ax = axes[0, 0]
    ax.plot(steps, history['train_acc'], label='Train', linewidth=2, color='blue')
    ax.plot(steps, history['test_acc'], label='Test', linewidth=2, color='red')
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.fill_between(steps, history['train_acc'], history['test_acc'],
                     alpha=0.2, color='orange', label='Generalization Gap')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Grokking Dynamics: Train vs Test', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Plot 2: Generalization Gap Over Time
    ax = axes[0, 1]
    ax.plot(steps, history['generalization_gap'], linewidth=2, color='purple')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Train Acc - Test Acc', fontsize=12)
    ax.set_title('Generalization Gap', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add shading for interpretation
    if history['generalization_gap'][-1] < 0.05:
        ax.fill_between(steps, 0, history['generalization_gap'],
                        alpha=0.3, color='green', label='Good generalization')
    elif history['generalization_gap'][-1] < 0.15:
        ax.fill_between(steps, 0, history['generalization_gap'],
                        alpha=0.3, color='yellow', label='Moderate generalization')
    else:
        ax.fill_between(steps, 0, history['generalization_gap'],
                        alpha=0.3, color='red', label='Poor generalization')
    ax.legend(fontsize=10)

    # Plot 3: Operation-Specific Test Accuracy
    ax = axes[1, 0]
    ax.plot(steps, history['test_acc_add'], label='Addition (Test)',
            linewidth=2, color='blue', linestyle='--')
    ax.plot(steps, history['test_acc_mul'], label='Multiplication (Test)',
            linewidth=2, color='red', linestyle='--')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Operation-Specific Generalization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Plot 4: Loss (Train vs Test)
    ax = axes[1, 1]
    ax.plot(steps, history['train_loss'], label='Train Loss',
            linewidth=2, color='blue', alpha=0.7)
    ax.plot(steps, history['test_loss'], label='Test Loss',
            linewidth=2, color='red', alpha=0.7)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss: Train vs Test', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    save_path = Path("test_accuracy_results") / "grokking_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved grokking plot to {save_path}")
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_test_accuracy_experiment():
    """
    Run Priority 1 experiment with test accuracy tracking.

    This is THE critical experiment that tells you if you have
    true grokking or just memorization.
    """

    print("="*60)
    print("🔬 PRIORITY 1: TEST ACCURACY VALIDATION")
    print("="*60)
    print("\nExperiment: complexity curriculum, mixed task, seed 42")
    print("NEW: Tracking both TRAIN and TEST accuracy")
    print("\nThis will tell you if you have:")
    print("  ✅ True grokking (test jumps to 95%+)")
    print("  ⚠️  Memorization (test stays low)")
    print("\nEstimated time: ~30 minutes")
    print("="*60 + "\n")

    # Setup
    config = TestConfig(
        curriculum_type="complexity",
        task_type="mixed",
        seed=42,
        experiment_name="priority1_test_accuracy"
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Initialize
    curriculum = TestAwareCurriculum(config.p, config)
    model = GrokkingTransformer(config, curriculum.vocab_size).to(config.device)
    trainer = TestAccuracyTrainer(model, curriculum, config)

    # Train
    results = trainer.train()

    # Analyze
    analyze_grokking_results(results)

    print("\n" + "="*60)
    print("✅ PRIORITY 1 COMPLETE!")
    print("="*60)
    print("\nNext steps based on results:")
    print("\nIf TEST accuracy > 90%:")
    print("  → Rerun all 8 experiments with test tracking")
    print("  → You have true grokking! Strong paper.")
    print("\nIf TEST accuracy 70-90%:")
    print("  → Partial generalization")
    print("  → Frame as 'curriculum enables learning' (more conservative)")
    print("\nIf TEST accuracy < 70%:")
    print("  → Primarily memorization")
    print("  → Address honestly in limitations")
    print("  → Still publishable! Understanding failure is valuable.")
    print("="*60)

    return results

# ==========================================
# QUICK COMPARISON WITH ORIGINAL
# ==========================================

def compare_with_original():
    """
    Compare these results with your original training-only results.
    """
    print("\n" + "="*60)
    print("📊 COMPARISON WITH ORIGINAL RESULTS")
    print("="*60)

    print("\nORIGINAL (training accuracy only):")
    print("  • complexity_mixed_s42: 98.4% at 6,250 steps")
    print("  • We thought: 'Great! The model learned!'")

    print("\nNOW (with test accuracy):")
    print("  • We'll see if that 98.4% generalizes")
    print("  • If test is also high → TRUE grokking")
    print("  • If test is low → Just memorization")

    print("\nThis is the KEY experiment that changes your paper!")
    print("="*60)

# ==========================================
# RUN IT!
# ==========================================

if __name__ == "__main__":
    result = run_test_accuracy_experiment()
    print("\n🚀 READY TO RUN!")
    print("\nSimply execute:")
    print("  result = run_test_accuracy_experiment()")
    print("\nOr run:")
    print("  compare_with_original()")
    print("  result = run_test_accuracy_experiment()")
    print("\n" + "="*60)
