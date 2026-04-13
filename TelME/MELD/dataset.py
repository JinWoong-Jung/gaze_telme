import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class meld_dataset(Dataset):
    def __init__(self, data, gaze_pkl: str = None):
        """
        Args:
            data:      list of sessions from preprocessing()
            gaze_pkl:  path to features/gaze/{split}.pkl.
                       When None (default) gaze is disabled — original behaviour.
        """
        self.emoList = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.session_dataset = data

        self.use_gaze = False
        self.gaze_cache: dict = {}
        if gaze_pkl is not None:
            with open(gaze_pkl, "rb") as f:
                self.gaze_cache = pickle.load(f)
            self.use_gaze = True

    def __len__(self):
        return len(self.session_dataset)

    def __getitem__(self, idx):
        session = self.session_dataset[idx]
        if not self.use_gaze:
            return session

        # Attach per-utterance gaze vecs; last turn's (dia, utt) is used for
        # the label utterance — we annotate every turn for completeness.
        session_with_gaze = []
        for turn in session:
            speaker, utt, video_path, emotion = turn
            # Parse dia/utt ids from video_path filename  e.g. dia3_utt7.mp4
            fname = Path(video_path).stem   # "dia3_utt7"
            try:
                parts = fname.split("_")
                d = int(parts[0].replace("dia", ""))
                u = int(parts[1].replace("utt", ""))
            except Exception:
                d, u = -1, -1
            gaze_vec = self.gaze_cache.get(
                (d, u), np.zeros(6, dtype=np.float32)
            )
            session_with_gaze.append([speaker, utt, video_path, emotion, gaze_vec])
        return session_with_gaze