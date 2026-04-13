"""Phase 4 — Unit tests for gaze-augmented dataloader.

Tests:
  1. use_gaze=False: collate output is a 5-tuple (original behaviour)
  2. use_gaze=True:  collate output is a 6-tuple with gaze tensor of shape (B, 6)
  3. Batch dtypes and devices are as expected
  4. Zero-vector fallback for missing gaze keys

Run:
    python -m pytest tests/test_dataloader.py -v
"""

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "TelME" / "MELD"))

from dataset import meld_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dummy_session(video_path: str, emotion: str = "neutral", n_turns: int = 3):
    """Build a minimal session list (no gaze attached yet — dataset adds it)."""
    session = []
    for i in range(n_turns):
        speaker   = i % 2
        utt       = f"Hello turn {i}"
        session.append([speaker, utt, video_path, emotion])
    return [session]   # list-of-sessions


def _make_gaze_pkl(tmp_dir: Path, keys: list) -> str:
    gaze_dict = {k: np.random.randn(6).astype(np.float32) for k in keys}
    pkl_path = tmp_dir / "train.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(gaze_dict, f)
    return str(pkl_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMeldDatasetNoGaze:
    def test_len(self, tmp_path):
        sessions = _make_dummy_session("dia0_utt0.mp4") * 5
        ds = meld_dataset(sessions)
        assert len(ds) == 5

    def test_getitem_no_gaze(self, tmp_path):
        sessions = _make_dummy_session("dia0_utt0.mp4")
        ds = meld_dataset(sessions)
        item = ds[0]
        # Each turn: [speaker, utt, video_path, emotion] — no gaze
        assert len(item[0]) == 4, "Expected 4-element turn tuple without gaze"

    def test_use_gaze_flag_false(self, tmp_path):
        ds = meld_dataset(_make_dummy_session("dia0_utt0.mp4"))
        assert ds.use_gaze is False


class TestMeldDatasetWithGaze:
    def test_use_gaze_flag_true(self, tmp_path):
        pkl = _make_gaze_pkl(tmp_path, [(0, 0)])
        ds  = meld_dataset(_make_dummy_session("dia0_utt0.mp4"), gaze_pkl=pkl)
        assert ds.use_gaze is True

    def test_getitem_has_gaze_vec(self, tmp_path):
        pkl = _make_gaze_pkl(tmp_path, [(0, 0)])
        ds  = meld_dataset(_make_dummy_session("dia0_utt0.mp4"), gaze_pkl=pkl)
        item = ds[0]
        # Each turn should now be 5-element
        for turn in item:
            assert len(turn) == 5, "Expected 5-element turn with gaze"
            gaze_vec = turn[4]
            assert isinstance(gaze_vec, np.ndarray)
            assert gaze_vec.shape == (6,)

    def test_missing_key_falls_back_to_zero(self, tmp_path):
        # Gaze pkl exists but doesn't contain key (5, 99)
        pkl = _make_gaze_pkl(tmp_path, [(0, 0)])
        # Session uses a path that would parse to (5, 99)
        sessions = _make_dummy_session("dia5_utt99.mp4")
        ds = meld_dataset(sessions, gaze_pkl=pkl)
        item = ds[0]
        gaze_vec = item[-1][4]  # last turn's gaze
        assert np.allclose(gaze_vec, 0.0), "Missing key should yield zero vector"

    def test_gaze_vec_dtype(self, tmp_path):
        pkl = _make_gaze_pkl(tmp_path, [(3, 7)])
        ds  = meld_dataset(_make_dummy_session("dia3_utt7.mp4"), gaze_pkl=pkl)
        gaze_vec = ds[0][-1][4]
        assert gaze_vec.dtype == np.float32


class TestCollate:
    """Test that make_batchs produces correct output tuples.

    NOTE: make_batchs calls librosa, cv2, and HuggingFace processors which
    need actual media files.  We mock those calls for fast unit testing.
    """

    def _mock_batch(self, n: int, use_gaze: bool):
        """Build minimal fake batch items matching dataset output format."""
        sessions = []
        for _ in range(n):
            session = []
            turn = [0, "hello", "dia0_utt0.mp4", "neutral"]
            if use_gaze:
                turn.append(np.zeros(6, dtype=np.float32))
            session.append(turn)
            sessions.append(session)
        return sessions

    def test_batch_tuple_len_no_gaze(self, monkeypatch):
        """Without gaze, batch should be a 5-tuple."""
        sys.path.insert(0, str(_ROOT / "TelME" / "MELD"))
        from utils import make_batchs
        import utils as meld_utils

        # Patch heavy I/O
        monkeypatch.setattr(meld_utils, "get_audio",
                            lambda *a, **kw: torch.zeros(1412))
        monkeypatch.setattr(meld_utils, "get_video",
                            lambda *a, **kw: torch.zeros(8, 3, 224, 224))
        monkeypatch.setattr("librosa.load", lambda *a, **kw: (np.zeros(100), 16000))
        monkeypatch.setattr("librosa.get_duration", lambda **kw: 5.0)

        batch = self._mock_batch(2, use_gaze=False)
        out   = make_batchs(batch)
        assert len(out) == 5, f"Expected 5-tuple, got {len(out)}"

    def test_batch_tuple_len_with_gaze(self, monkeypatch):
        """With gaze, batch should be a 6-tuple."""
        sys.path.insert(0, str(_ROOT / "TelME" / "MELD"))
        from utils import make_batchs
        import utils as meld_utils

        monkeypatch.setattr(meld_utils, "get_audio",
                            lambda *a, **kw: torch.zeros(1412))
        monkeypatch.setattr(meld_utils, "get_video",
                            lambda *a, **kw: torch.zeros(8, 3, 224, 224))
        monkeypatch.setattr("librosa.load", lambda *a, **kw: (np.zeros(100), 16000))
        monkeypatch.setattr("librosa.get_duration", lambda **kw: 5.0)

        batch = self._mock_batch(2, use_gaze=True)
        out   = make_batchs(batch)
        assert len(out) == 6, f"Expected 6-tuple, got {len(out)}"

        gaze_t = out[5]
        assert isinstance(gaze_t, torch.Tensor)
        assert gaze_t.shape == (2, 6), f"Gaze tensor shape: {gaze_t.shape}"
        assert gaze_t.dtype == torch.float32
