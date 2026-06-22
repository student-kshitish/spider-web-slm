import math
import torch
from torch.optim.lr_scheduler import LambdaLR

# -------------------------
# LEARNING RATE SCHEDULER
# -------------------------
def get_cosine_warmup_scheduler(optimizer, config):
    """
    The 'Reasoning' LR Scheduler.
    Uses a Linear Warmup to stabilize the Lorenz Router, 
    followed by a Cosine Decay to freeze semantic logic into the SRM.
    """
    warmup_steps = config.train.warmup_steps
    total_steps = config.train.steps
    lr_min_ratio = config.train.lr_min / config.train.lr # Usually 0.1 or 0.01

    def lr_lambda(current_step: int):
        # 1. Linear Warmup Phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 2. Cosine Decay Phase
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Clamp progress at 1.0 to prevent cosine from going back up
        progress = min(1.0, progress)
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between 1.0 (base_lr) and lr_min_ratio
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# -------------------------
# TEMPERATURE SCHEDULER (The Chaos Cooler)
# -------------------------
class TemperatureScheduler:
    """
    Annelas the Gumbel-Softmax temperature (tau).
    
    Why this surpasses Big LLMs:
    - High Temp (Start): The Lorenz Router explores ALL nodes in the web.
    - Low Temp (End): The Router locks into the single most logical 'hop'.
    """
    def __init__(self, config):
        self.start_temp = config.routing.temp_start
        self.end_temp = config.routing.temp_end
        self.total_steps = config.routing.anneal_steps
        
        # Calculate the constant decay rate for exponential cooling
        # This is smoother and more 'physical' for a chaotic system
        self.decay_rate = math.log(self.start_temp / self.end_temp) / self.total_steps

    def get_temp(self, step):
        """Returns the current tau for the forward pass."""
        if step >= self.total_steps:
            return self.end_temp
        
        # Exponential Decay: T_t = T_0 * e^(-kt)
        temp = self.start_temp * math.exp(-self.decay_rate * step)
        
        return max(temp, self.end_temp)