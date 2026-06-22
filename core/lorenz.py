import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

class LorenzRouter(nn.Module):
    """
    The 'Spider's Brain': A Jacobian-stabilized chaotic router.
    Uses Lorenz-63 dynamics to map context + memory into a 3-way routing decision.
    """
    def __init__(self, config):
        super().__init__()
        self.cfg = config.lorenz
        self.dim = config.model.dim
        
        # Projections with Spectral Normalization (Modern PyTorch implementation)
        # Maps hidden state to Lorenz space (x, y, z)
        self.proj_in = spectral_norm(nn.Linear(self.dim, 3, bias=False))
        # Maps chaotic trajectory back to routing logits [Stay, Jump_In, Jump_Out]
        self.proj_out = spectral_norm(nn.Linear(3, 3, bias=False))

        # Stabilization Parameters
        self.eps = nn.Parameter(torch.tensor([self.cfg.eps_init]))
        self.n_steps = 5  # Unroll depth: Higher = more chaotic sensitivity
        
    def _lorenz_step(self, xyz: torch.Tensor) -> torch.Tensor:
        """One-step Euler integration with hard-clamping."""
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        
        dx = self.cfg.sigma * (y - x)
        dy = x * (self.cfg.rho - z) - y
        dz = x * y - self.cfg.beta * z
        
        deriv = torch.stack([dx, dy, dz], dim=-1)
        xyz_new = xyz + self.cfg.dt * deriv
        
        # Secure the trajectory within stable bounds
        return xyz_new.clamp(-self.cfg.clamp, self.cfg.clamp)

    def forward(
        self, 
        h: torch.Tensor, 
        M_t: torch.Tensor, 
        tau: float = 1.0, 
        hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Current hidden state (B, dim)
            M_t: Temporal SRM Memory (B, slots, dim)
            tau: Gumbel temperature
            hard: If True, returns discrete one-hot paths
        """
        # 1. 🛡️ Jacobian Guard
        # We nudge 'h' by a tiny fraction of its chaotic derivative.
        # This keeps the token identity intact while allowing it to 'feel' the chaos.
        xyz0 = self.proj_in(h)
        f_h = self.proj_out(xyz0) 
        
        # Map chaos back to model space for the residual nudge
        h_guarded = h + self.eps * torch.tanh(
            F.linear(f_h, self.proj_in.weight.t()) 
        )

        # 2. 🧠 Memory Integration
        # We blend the current token with a summary of the Solar Ring Memory.
        # This ensures the 'Spider' routes based on sequence history.
        mem_ctx = M_t.mean(dim=1) # Global context from SRM
        xyz = self.proj_in(h_guarded + 0.1 * mem_ctx)

        # 3. 🌀 Chaotic Traversal (Unrolling the ODE)
        # 5 steps allow the chaotic butterfly effect to actually manifest.
        for _ in range(self.n_steps):
            xyz = self._lorenz_step(xyz)

        # 4. 🧭 Routing Decision
        logits = self.proj_out(xyz)

        # 5. 🎲 Gumbel-Softmax
        # During training, this is soft; during inference, it becomes a hard jump.
        routing = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
        return routing, h_guarded