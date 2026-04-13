"""Phase 5 — Unit tests for VideoGazeStudent and GazeProjector.

Tests:
  1. GazeProjector: output shape (B, 768), grad flow
  2. VideoGazeStudent with each fusion type: output shapes
  3. Forward + backward pass (grad check for new parameters)
  4. Fusion type validation

Run (no GPU needed — uses random weights / mocked encoder):
    python -m pytest tests/test_video_gaze_student.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.gaze_projector import GazeProjector


# ---------------------------------------------------------------------------
# GazeProjector tests
# ---------------------------------------------------------------------------

class TestGazeProjector:
    def test_output_shape(self):
        proj = GazeProjector(in_dim=6, hidden=64, out_dim=768)
        x    = torch.randn(4, 6)
        out  = proj(x)
        assert out.shape == (4, 768)

    def test_custom_dims(self):
        proj = GazeProjector(in_dim=10, hidden=32, out_dim=256)
        x    = torch.randn(8, 10)
        assert proj(x).shape == (8, 256)

    def test_grad_flows(self):
        proj = GazeProjector()
        x    = torch.randn(2, 6, requires_grad=True)
        loss = proj(x).sum()
        loss.backward()
        assert x.grad is not None
        for p in proj.parameters():
            assert p.grad is not None

    def test_in_dim_out_dim_properties(self):
        proj = GazeProjector(in_dim=6, out_dim=768)
        assert proj.in_dim  == 6
        assert proj.out_dim == 768


# ---------------------------------------------------------------------------
# Fusion module tests (no real TimeSformer needed)
# ---------------------------------------------------------------------------

class _FakeVideoEncoder(nn.Module):
    """Minimal stub for TimeSformerModel to avoid downloading weights."""
    class _Config:
        hidden_size = 768
    config = _Config()

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 768)

    def forward(self, x):
        B = x.shape[0]
        out = torch.zeros(B, 1, 768)
        return MagicMock(last_hidden_state=out)


def _make_student(fusion_type: str = "gated", lam: float = 0.3,
                  cls_num: int = 7) -> "VideoGazeStudent":
    """Build student with mocked encoder (no pretrained download)."""
    from models.video_gaze_student import VideoGazeStudent
    with patch("models.video_gaze_student.TimesformerModel") as MockTS:
        fake_enc = _FakeVideoEncoder()
        MockTS.from_pretrained.return_value = fake_enc
        student = VideoGazeStudent.__new__(VideoGazeStudent)
        nn.Module.__init__(student)
        from models.gaze_projector import GazeProjector
        from models.video_gaze_student import _build_fusion
        student.video_encoder = fake_enc
        student.gaze_projector = GazeProjector(in_dim=6, hidden=128, out_dim=768)
        student.fusion     = _build_fusion(fusion_type, hidden_dim=768, lam=lam)
        student.classifier = nn.Linear(768, cls_num)
        student.hidden_dim  = 768
        student.fusion_type = fusion_type
    return student


class TestVideoGazeStudentShapes:
    @pytest.mark.parametrize("fusion_type", ["add", "concat_proj", "gated"])
    def test_output_shape(self, fusion_type):
        student = _make_student(fusion_type)
        B = 3
        video = torch.randn(B, 8, 3, 224, 224)  # TimeSformer pixel_values shape
        gaze  = torch.randn(B, 6)
        hidden, logit = student(video, gaze)
        assert hidden.shape == (B, 768), f"[{fusion_type}] hidden: {hidden.shape}"
        assert logit.shape  == (B, 7),   f"[{fusion_type}] logit:  {logit.shape}"

    @pytest.mark.parametrize("fusion_type", ["add", "concat_proj", "gated"])
    def test_grad_flow(self, fusion_type):
        """New parameters (projector + fusion) must receive gradients."""
        student = _make_student(fusion_type)
        B = 2
        video = torch.randn(B, 8, 3, 224, 224)
        gaze  = torch.randn(B, 6)
        _, logit = student(video, gaze)
        loss = logit.sum()
        loss.backward()
        for name, p in student.gaze_projector.named_parameters():
            assert p.grad is not None, f"No grad for projector.{name}"
        for name, p in student.fusion.named_parameters():
            assert p.grad is not None, f"No grad for fusion.{name}"

    def test_invalid_fusion_type(self):
        from models.video_gaze_student import _build_fusion
        with pytest.raises(ValueError, match="Unknown fusion_type"):
            _build_fusion("unknown_mode")

    def test_add_fusion_lambda(self):
        """Add fusion output should equal video + λ*gaze."""
        from models.video_gaze_student import _AddFusion
        lam  = 0.5
        f    = _AddFusion(lam=lam)
        v    = torch.ones(2, 768)
        g    = torch.ones(2, 768)
        out  = f(v, g)
        expected = v + lam * g
        assert torch.allclose(out, expected)

    @pytest.mark.parametrize("B", [1, 4, 8])
    def test_batch_sizes(self, B):
        student = _make_student("gated")
        video = torch.randn(B, 8, 3, 224, 224)
        gaze  = torch.randn(B, 6)
        hidden, logit = student(video, gaze)
        assert hidden.shape[0] == B
        assert logit.shape[0]  == B
