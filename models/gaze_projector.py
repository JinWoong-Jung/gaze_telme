"""GazeProjector — projects R^6 gaze features into the model hidden space.

Architecture:
  LayerNorm(6) → Linear(6→128) → GELU → Dropout(0.1) → Linear(128→768)

Hidden dimension 768 matches both TimeSformer and RoBERTa-large's W projection.
"""

import torch
import torch.nn as nn


class GazeProjector(nn.Module):
    """Map utterance-level gaze_vec ∈ R^in_dim  →  hidden ∈ R^out_dim.

    Args:
        in_dim:  input gaze feature dimension (default 6)
        hidden:  intermediate MLP width (default 128)
        out_dim: output dimension, must match video/text hidden size (default 768)
        p:       dropout probability (default 0.1)
    """

    def __init__(self, in_dim: int = 6, hidden: int = 128,
                 out_dim: int = 768, p: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )
        self._out_dim = out_dim
        self._in_dim  = in_dim

    def forward(self, gaze: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gaze: (B, in_dim)  float32
        Returns:
            projected: (B, out_dim)  float32
        """
        return self.mlp(gaze)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @property
    def in_dim(self) -> int:
        return self._in_dim
