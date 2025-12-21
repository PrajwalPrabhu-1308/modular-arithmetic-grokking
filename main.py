import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import json
import logging

# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class ExperimentConfig:
    """Centralized configuration for reproducibility."""
    # Model Architecture
    p: int = 113                    # Prime modulus
    dim: int = 128                  # Embedding dimension
    n_heads: int = 4                # Attention heads
    n_layers: int = 2               # Transformer layers
    dropout: float = 0.0            # Dropout rate (0 for grokking studies)

    # Training Hyperparameters
    learning_rate: float = 1e-3     # Base learning rate
    weight_decay: float = 1.0       # Critical for grokking
    max_steps: int = 25000          # Training iterations
    batch_size: int = 512           # Samples per batch
    warmup_steps: int = 100         # LR warmup period

    # Curriculum Settings
    use_curriculum: bool = True     # Enable ZPD-based curriculum
    curriculum_threshold: float = 0.95  # Accuracy threshold for advancement
    curriculum_window: int = 20     # Steps to average for stability

    # Positional Encoding
    use_alibi: bool = True          # Use ALiBi instead of learned PE
    max_seq_len: int = 64           # Maximum sequence length

    # Logging & Checkpointing
    log_interval: int = 100         # Steps between evaluations
    snapshot_interval: int = 1000   # Steps between PCA snapshots
    save_dir: str = "grokking_output"

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42                  # Random seed

    def __post_init__(self):
        """Validate configuration."""
        assert self.p > 2, "Prime modulus must be > 2"
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.curriculum_threshold < 1.0, "Threshold must be < 1.0"

# ==========================================
# LOGGING SETUP
# ==========================================

def setup_logging(config: ExperimentConfig) -> logging.Logger:
    """Configure structured logging."""
    log_dir = Path(config.save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "experiment.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==========================================
# CURRICULUM ENVIRONMENT
# ==========================================

class ModularArithmeticCurriculum:
    """
    Complexity-based curriculum for modular arithmetic.

    Instead of range-based progression (which creates a "linear trap"),
    this curriculum focuses on wrapping complexity:
    - Level 0: Easy wrapping (results near 0 or P)
    - Level 1: Medium wrapping (all operations, balanced sampling)
    - Level 2: Hard cases (operations requiring deep modular understanding)
    """

    def __init__(self, p: int, config: ExperimentConfig):
        self.p = p
        self.config = config
        self.level = 0
        self.max_level = 2

        # Special token IDs
        self.delim_token = p        # "="
        self.add_token = p + 1      # "+"
        self.mul_token = p + 2      # "*"
        self.vocab_size = p + 3

        # Curriculum tracking
        self.accuracy_history: List[float] = []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized curriculum with P={p}, vocab_size={self.vocab_size}")

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training batch based on current curriculum level.

        Returns:
            x: Input tensor [batch_size, 4] with format [a, b, op, delim]
            y: Target tensor [batch_size] with correct answers mod P
        """
        device = self.config.device

        if not self.config.use_curriculum or self.level == self.max_level:
            # Full distribution: uniform sampling
            a = torch.randint(0, self.p, (batch_size,), device=device)
            b = torch.randint(0, self.p, (batch_size,), device=device)

        elif self.level == 0:
            # Level 0: Focus on wrapping cases
            # Sample pairs where (a+b) or (a*b) wrap around modulus
            # Strategy: Sample from regions where wrapping is guaranteed
            a = torch.randint(self.p // 2, self.p, (batch_size,), device=device)
            b = torch.randint(self.p // 2, self.p, (batch_size,), device=device)

        else:  # level == 1
            # Level 1: Mixed difficulty - 70% full range, 30% wrapping focus
            n_full = int(batch_size * 0.7)
            n_wrap = batch_size - n_full

            a_full = torch.randint(0, self.p, (n_full,), device=device)
            b_full = torch.randint(0, self.p, (n_full,), device=device)

            a_wrap = torch.randint(self.p // 2, self.p, (n_wrap,), device=device)
            b_wrap = torch.randint(self.p // 2, self.p, (n_wrap,), device=device)

            a = torch.cat([a_full, a_wrap])
            b = torch.cat([b_full, b_wrap])

        # 50/50 split between addition and multiplication
        is_mul = torch.rand(batch_size, device=device) > 0.5
        ops = torch.where(is_mul, self.mul_token, self.add_token)

        # Compute targets (always mod P)
        y_add = (a + b) % self.p
        y_mul = (a * b) % self.p
        y = torch.where(is_mul, y_mul, y_add)

        # Construct input: [a, b, op, delim]
        delim = torch.full_like(a, self.delim_token)
        x = torch.stack([a, b, ops, delim], dim=1)

        return x, y

    def update_progress(self, accuracy: float) -> bool:
        """
        Update curriculum based on current accuracy.

        Returns:
            True if level advanced, False otherwise
        """
        if not self.config.use_curriculum or self.level >= self.max_level:
            return False

        self.accuracy_history.append(accuracy)

        # Maintain sliding window
        if len(self.accuracy_history) > self.config.curriculum_window:
            self.accuracy_history.pop(0)

        # Check if ready to advance
        if len(self.accuracy_history) >= self.config.curriculum_window:
            avg_acc = np.mean(self.accuracy_history)

            if avg_acc >= self.config.curriculum_threshold:
                self.level += 1
                self.accuracy_history.clear()
                self.logger.info(f"🚀 CURRICULUM ADVANCED TO LEVEL {self.level} (avg_acc={avg_acc:.4f})")
                return True

        return False

# ==========================================
# MODEL ARCHITECTURE
# ==========================================

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with optional ALiBi positional bias.
    Implements proper GPT-style autoregressive attention.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        assert config.dim % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dim = config.dim

        # QKV projection (weight tying opportunity)
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Register causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

        # ALiBi slopes (if enabled)
        if config.use_alibi:
            self.register_buffer("alibi_slopes", self._compute_alibi_slopes(config.n_heads))
        else:
            self.alibi_slopes = None

    @staticmethod
    def _compute_alibi_slopes(n_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes using geometric sequence.
        Formula: m_h = 2^(-8h/n) for h in [1, n]
        """
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-8 / n)
                ratio = start
                return [start * (ratio ** i) for i in range(n)]

            if n & (n - 1) == 0:  # Power of 2
                return get_slopes_power_of_2(n)
            else:
                # Nearest power of 2
                closest_power = 2 ** np.floor(np.log2(n))
                slopes = get_slopes_power_of_2(int(closest_power))
                slopes += get_slopes(int(2 * closest_power))[0::2][:n - int(closest_power)]
                return slopes

        return torch.tensor(get_slopes(n_heads), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))

        # Apply causal mask
        att = att.masked_fill(self.causal_mask[:T, :T], float('-inf'))

        # Apply ALiBi bias (additive)
        if self.alibi_slopes is not None:
            # Create position distance matrix
            positions = torch.arange(T, device=x.device)
            alibi_bias = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
            # Apply slopes: (n_heads, 1, 1) * (T, T) -> (n_heads, T, T)
            alibi_bias = -self.alibi_slopes.view(-1, 1, 1) * alibi_bias.unsqueeze(0)
            att = att + alibi_bias.unsqueeze(0)  # Broadcast to (B, nh, T, T)

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Aggregate values
        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)

class TransformerBlock(nn.Module):
    """GPT-style transformer block with pre-norm architecture."""

    def __init__(self, config: ExperimentConfig):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GrokkingTransformer(nn.Module):
    """
    Causal transformer for modular arithmetic with weight tying.

    Architecture follows GPT conventions with careful attention to
    embedding space organization for grokking phenomena.
    """

    def __init__(self, config: ExperimentConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.p = config.p
        self.vocab_size = vocab_size

        # Token embeddings (input)
        self.token_emb = nn.Embedding(vocab_size, config.dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.dim)

        # Output projection uses weight tying with input embeddings
        # Only project to numeric tokens [0, P-1]
        # This is mathematically correct: outputs are always numbers mod P

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        def _init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T] with token indices

        Returns:
            logits: Output logits [B, T, P] over numeric tokens
        """
        B, T = x.shape

        # Token embeddings
        x = self.token_emb(x)  # (B, T, dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Output projection with weight tying
        # Use only numeric token embeddings [0, P-1] for output
        logits = F.linear(x, self.token_emb.weight[:self.p])  # (B, T, P)

        return logits

    def get_embedding_weights(self) -> np.ndarray:
        """Extract numeric embeddings for analysis."""
        return self.token_emb.weight[:self.p].detach().cpu().numpy()

# ==========================================
# TRAINING & EVALUATION
# ==========================================

class GrokkingTrainer:
    """Manages training loop with logging and checkpointing."""

    def __init__(
        self,
        model: GrokkingTransformer,
        curriculum: ModularArithmeticCurriculum,
        config: ExperimentConfig,
        logger: logging.Logger
    ):
        self.model = model
        self.curriculum = curriculum
        self.config = config
        self.logger = logger

        # Optimizer with weight decay (critical for grokking)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )

        # Metrics tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'curriculum_level': [],
            'learning_rate': []
        }

        # Setup output directories
        self.output_dir = Path(config.save_dir)
        self.snapshot_dir = self.output_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Single training step."""
        self.model.train()

        # Forward pass
        logits = self.model(x)  # (B, T, P)

        # We only care about the prediction at the last position
        logits_last = logits[:, -1, :]  # (B, P)

        # Cross-entropy loss
        loss = F.cross_entropy(logits_last, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional, but helps stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            pred = logits_last.argmax(dim=-1)
            acc = (pred == y).float().mean().item()

        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, n_batches: int = 10) -> Tuple[float, float]:
        """Evaluate on held-out distribution."""
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0

        for _ in range(n_batches):
            x, y = self.curriculum.get_batch(self.config.batch_size)
            logits = self.model(x)
            logits_last = logits[:, -1, :]

            loss = F.cross_entropy(logits_last, y)
            pred = logits_last.argmax(dim=-1)
            acc = (pred == y).float().mean().item()

            total_loss += loss.item()
            total_acc += acc

        return total_loss / n_batches, total_acc / n_batches

    def save_snapshot(self, step: int, acc: float):
        """Save PCA visualization of embedding space."""
        embeddings = self.model.get_embedding_weights()  # (P, dim)

        # PCA to 2D
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

        scatter = ax.scatter(
            proj[:, 0], proj[:, 1],
            c=np.arange(self.config.p),
            cmap='hsv',
            s=50,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )

        # Connect points to show ring structure
        proj_wrapped = np.vstack([proj, proj[0:1]])  # Close the loop
        ax.plot(proj_wrapped[:, 0], proj_wrapped[:, 1],
                'k-', alpha=0.2, linewidth=0.5)

        plt.colorbar(scatter, ax=ax, label='Token Value')
        ax.set_title(f'Step {step} | Level {self.curriculum.level} | Acc {acc:.3f}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.snapshot_dir / f'step_{step:06d}.png', dpi=100)
        plt.close()

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'history': self.history
        }

        path = self.output_dir / f'checkpoint_step_{step}.pt'
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {asdict(self.config)}")

        pbar = tqdm(range(self.config.max_steps), desc="Training")

        for step in pbar:
            # Get batch
            x, y = self.curriculum.get_batch(self.config.batch_size)

            # Training step
            loss, acc = self.train_step(x, y)

            # Update learning rate
            self.scheduler.step()

            # Logging
            if step % self.config.log_interval == 0:
                self.history['train_loss'].append(loss)
                self.history['train_acc'].append(acc)
                self.history['curriculum_level'].append(self.curriculum.level)
                self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

                # Update curriculum
                leveled_up = self.curriculum.update_progress(acc)

                # Save snapshot on level up or periodically
                if leveled_up or step % self.config.snapshot_interval == 0:
                    self.save_snapshot(step, acc)

                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{acc:.3f}',
                    'lvl': self.curriculum.level,
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        # Final checkpoint
        self.save_checkpoint(self.config.max_steps)
        self.logger.info("Training complete!")

        return self.history

# ==========================================
# VISUALIZATION
# ==========================================

def plot_training_curves(history: Dict, save_path: Path):
    """Plot comprehensive training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    steps = np.arange(len(history['train_acc'])) * 100  # Log interval

    # Accuracy
    axes[0, 0].plot(steps, history['train_acc'], linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Training Accuracy (Grokking Curve)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    # Loss
    axes[0, 1].plot(steps, history['train_loss'], linewidth=2, color='red')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Curriculum Level
    axes[1, 0].plot(steps, history['curriculum_level'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Curriculum Level')
    axes[1, 0].set_title('Curriculum Progression')
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(steps, history['learning_rate'], linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ==========================================
# MAIN EXPERIMENT
# ==========================================

def run_experiment(config: Optional[ExperimentConfig] = None):
    """Execute complete grokking experiment."""
    if config is None:
        config = ExperimentConfig()

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("GROKKING EXPERIMENT: Modular Arithmetic")
    logger.info("=" * 60)

    # Initialize components
    curriculum = ModularArithmeticCurriculum(config.p, config)
    model = GrokkingTransformer(config, curriculum.vocab_size).to(config.device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    trainer = GrokkingTrainer(model, curriculum, config, logger)

    # Train
    history = trainer.train()

    # Save final visualizations
    output_dir = Path(config.save_dir)
    plot_training_curves(history, output_dir / "training_curves.png")

    # Save history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    logger.info("Experiment complete! 🎉")

    return model, history

if __name__ == "__main__":
    # Run with default configuration
    model, history = run_experiment()

    print("\n" + "="*60)
    print("Experiment finished successfully!")
    print(f"Final accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Check 'grokking_output' directory for results")
    print("="*60)
