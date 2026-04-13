"""Phase 5 — Video + Gaze student model.

Wraps TelME's Student_Video with a GazeProjector and one of three
fusion strategies:

  add         : video_hidden + λ * gaze_hidden
  concat_proj : Linear(768+768 → 768) applied to [video; gaze]
  gated       : g = σ(W·[video; gaze]); out = g*video + (1-g)*gaze_hidden

The fused hidden is passed to the same classifier head as the original video
student, keeping the output signature identical: (hidden, logit).

Hidden dimension is fixed at 768 to match TimeSformer / RoBERTa-large.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root on path so we can import from models/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.gaze_projector import GazeProjector


# ---------------------------------------------------------------------------
# Fusion modules
# ---------------------------------------------------------------------------

class _AddFusion(nn.Module):
    """video_hidden + λ * gaze_hidden."""

    def __init__(self, lam: float = 0.3):
        super().__init__()
        self.lam = lam

    def forward(self, video: torch.Tensor, gaze: torch.Tensor) -> torch.Tensor:
        return video + self.lam * gaze


class _ConcatProjFusion(nn.Module):
    """Linear([video; gaze] → hidden)."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, video: torch.Tensor, gaze: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([video, gaze], dim=-1))


class _GatedFusion(nn.Module):
    """Soft gate: g = σ(W·[video; gaze]); out = g*video + (1-g)*gaze."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, video: torch.Tensor, gaze: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([video, gaze], dim=-1)))
        return g * video + (1.0 - g) * gaze


def _build_fusion(fusion_type: str, hidden_dim: int = 768,
                  lam: float = 0.3) -> nn.Module:
    if fusion_type == "add":
        return _AddFusion(lam=lam)
    if fusion_type == "concat_proj":
        return _ConcatProjFusion(hidden_dim=hidden_dim)
    if fusion_type == "gated":
        return _GatedFusion(hidden_dim=hidden_dim)
    raise ValueError(f"Unknown fusion_type: {fusion_type!r}. "
                     "Choose from ['add', 'concat_proj', 'gated']")


# ---------------------------------------------------------------------------
# VideoGazeStudent
# ---------------------------------------------------------------------------

class VideoGazeStudent(nn.Module):
    """TimeSformer video student with gaze injection.

    Args:
        video_model: HuggingFace model id or local path for TimeSformer
        cls_num:     number of emotion classes (7 for MELD)
        fusion_type: 'add' | 'concat_proj' | 'gated'
        fusion_lambda: λ used only by 'add' fusion
        gaze_in_dim:   input gaze feature dimension (default 6)
        gaze_hidden:   GazeProjector intermediate width (default 128)
        gaze_dropout:  GazeProjector dropout (default 0.1)
    """

    def __init__(
        self,
        video_model: str = "facebook/timesformer-base-finetuned-k400",
        cls_num: int = 7,
        fusion_type: str = "gated",
        fusion_lambda: float = 0.3,
        gaze_in_dim: int = 6,
        gaze_hidden: int = 128,
        gaze_dropout: float = 0.1,
    ):
        super().__init__()

        from transformers import TimesformerModel
        self.video_encoder = TimesformerModel.from_pretrained(video_model)
        hidden_dim = self.video_encoder.config.hidden_size  # 768

        self.gaze_projector = GazeProjector(
            in_dim=gaze_in_dim,
            hidden=gaze_hidden,
            out_dim=hidden_dim,
            p=gaze_dropout,
        )

        self.fusion = _build_fusion(fusion_type, hidden_dim=hidden_dim,
                                    lam=fusion_lambda)

        self.classifier = nn.Linear(hidden_dim, cls_num)
        self.hidden_dim  = hidden_dim
        self.fusion_type = fusion_type

    def forward(
        self,
        video: torch.Tensor,   # (B, T, C, H, W)  — TimeSformer pixel_values
        gaze: torch.Tensor,    # (B, gaze_in_dim)
    ):
        """Return (fused_hidden, logit) matching TelME student interface."""
        video_out = self.video_encoder(video).last_hidden_state[:, 0, :]  # (B, 768)
        gaze_h    = self.gaze_projector(gaze)                              # (B, 768)
        fused     = self.fusion(video_out, gaze_h)                         # (B, 768)
        logit     = self.classifier(fused)                                 # (B, cls_num)
        return fused, logit
