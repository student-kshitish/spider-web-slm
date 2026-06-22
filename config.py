from dataclasses import dataclass, field
import torch


# -------------------------
# MODEL CONFIG
# -------------------------
@dataclass
class ModelConfig:
    dim: int = 96
    hidden_dim: int = 384
    num_rings: int = 4
    nodes_per_ring: int = 8
    vocab_size: int = 5000
    max_seq_len: int = 128
    rope_theta: float = 10000.0


# -------------------------
# MEMORY (SRM v2.1)
# -------------------------
@dataclass
class MemoryConfig:
    slots: int = 32
    alpha: float = 0.9
    beta: float = 0.1
    learned_init: bool = True


# -------------------------
# LORENZ / CHAOS
# -------------------------
@dataclass
class LorenzConfig:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 2.667
    dt: float = 0.005
    clamp: float = 15.0
    eps_init: float = 0.01


# -------------------------
# ROUTING
# -------------------------
@dataclass
class RoutingConfig:
    temp_start: float = 2.0
    temp_end: float = 0.1
    anneal_steps: int = 8000
    max_hops: int = 6


# -------------------------
# TRAINING (FINAL FIXED)
# -------------------------
@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 2.5e-4
    lr_min: float = 1e-5
    weight_decay: float = 0.001
    grad_clip: float = 1.0
    warmup_steps: int = 200
    steps: int = 30000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔥 REQUIRED BY train_main.py
    use_bf16: bool = True
    use_compile: bool = False   # set False first (compile can break early runs)

    # Optional dtype (safe default)
    dtype: torch.dtype = torch.bfloat16

    # -------------------------
    # LOSS WEIGHTS
    # -------------------------
    entropy_weight: float = 0.003
    balance_weight: float = 0.001


# -------------------------
# MAIN CONFIG
# -------------------------
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    lorenz: LorenzConfig = field(default_factory=LorenzConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> Config:
    return Config()