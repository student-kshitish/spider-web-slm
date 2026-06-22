import torch
import torch.nn as nn
import math


class SpiderWebRoPE(nn.Module):
    """
    3-Axis Polar-Sequence RoPE

    Splits d_model into:
    - Sequence (Temporal)
    - Angular (Node position)
    - Radial (Ring depth)
    """

    def __init__(self, config, base: float = 10000.0):
        super().__init__()

        d_model      = config.model.dim
        self.n_nodes = config.model.nodes_per_ring
        self.n_rings = config.model.num_rings

        # Split dimensions
        self.d_seq   = d_model // 2
        self.d_polar = d_model // 4
        self.base    = base

        # Frequency bands
        inv_freq_seq = 1.0 / (
            base ** (torch.arange(0, self.d_seq, 2).float() / self.d_seq)
        )

        inv_freq_polar = 1.0 / (
            base ** (torch.arange(0, self.d_polar, 2).float() / self.d_polar)
        )

        self.register_buffer("inv_freq_seq", inv_freq_seq)
        self.register_buffer("inv_freq_polar", inv_freq_polar)

    # -------------------------
    # ROTATION HELPERS
    # -------------------------
    def _rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def _apply_rotation(self, x, pos, inv_freq):
        """
        x: (B, T, d_sub)
        pos: (B, T)
        """

        freqs = torch.einsum("bt,f->btf", pos.float(), inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return x * cos + self._rotate_half(x) * sin

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, x, node_idx, ring_idx, seq_pos=None):
        """
        x: (B, T, D) or (B, D)
        node_idx: int or (B, T)
        ring_idx: int or (B, T)
        """

        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, T, D = x.shape
        device  = x.device

        # -------------------------
        # SEQUENCE (Temporal)
        # -------------------------
        if seq_pos is None:
            seq_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        x_seq = x[..., :self.d_seq]
        x_seq = self._apply_rotation(x_seq, seq_pos, self.inv_freq_seq)

        # -------------------------
        # ANGULAR (Node position)
        # -------------------------
        if isinstance(node_idx, (int, float)):
            theta = torch.full((B, T), node_idx, device=device)
        else:
            theta = node_idx

        # ✅ FIXED LINE
        theta = 2 * math.pi * theta / self.n_nodes

        x_ang = x[..., self.d_seq : self.d_seq + self.d_polar]
        x_ang = self._apply_rotation(x_ang, theta, self.inv_freq_polar)

        # -------------------------
        # RADIAL (Ring depth)
        # -------------------------
        if isinstance(ring_idx, (int, float)):
            radius = torch.full((B, T), ring_idx, device=device)
        else:
            radius = ring_idx

        radius = radius / max(self.n_rings - 1, 1)

        x_rad = x[..., self.d_seq + self.d_polar :]
        x_rad = self._apply_rotation(x_rad, radius, self.inv_freq_polar)

        # -------------------------
        # MERGE
        # -------------------------
        out = torch.cat([x_seq, x_ang, x_rad], dim=-1)

        return out.squeeze(1) if T == 1 else out