"""PyTorch Dataset for lip-reading: maps filtered subtitle entries to clip frames."""

import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_timestamp_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def ts_to_clip_key(ts: str) -> str:
    """Convert '00:00:24.519' → '00h00m24s519' to match crop_and_filter clip filenames."""
    return ts.replace(":", "h", 1).replace(":", "m", 1).replace(".", "s")


class LipReadingDataset(Dataset):
    """
    Loads mouth-region clips and their corresponding text labels.

    Expects:
      subtitle_path : filtered subtitle txt produced by crop_and_filter.py
                      format: [HH:MM:SS.mmm --> HH:MM:SS.mmm]  <text>
      clips_dir     : directory containing clip_NNNN_<ts>.mp4 files
    """

    def __init__(
        self,
        subtitle_path: str,
        clips_dir: str,
        frame_size: tuple[int, int] = (88, 88),
        max_frames: int = 100,
        vocab: list[str] | None = None,
    ):
        self.clips_dir = clips_dir
        self.frame_size = frame_size
        self.max_frames = max_frames

        self.entries = self._load_entries(subtitle_path)
        if not self.entries:
            raise ValueError(f"No matching clip/subtitle pairs found.\nSubtitle: {subtitle_path}\nClips dir: {clips_dir}")

        # Build or reuse vocabulary (index 0 reserved for CTC blank)
        if vocab is not None:
            self.vocab = vocab
        else:
            chars = set()
            for e in self.entries:
                chars.update(e["text"])
            self.vocab = ["<blank>"] + sorted(chars)

        self.char2idx = {c: i for i, c in enumerate(self.vocab)}

    def _load_entries(self, subtitle_path: str) -> list[dict]:
        pattern = re.compile(
            r"\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s+(.*)"
        )
        # Pre-index clip files by their embedded timestamp key
        clip_index: dict[str, str] = {}
        for fname in os.listdir(self.clips_dir):
            if fname.endswith(".mp4"):
                clip_index[fname] = os.path.join(self.clips_dir, fname)

        entries = []
        with open(subtitle_path, encoding="utf-8") as f:
            for line in f:
                m = pattern.match(line.strip())
                if not m:
                    continue
                start_ts, end_ts, text = m.group(1), m.group(2), m.group(3).strip()
                if not text:
                    continue

                key = ts_to_clip_key(start_ts)
                clip_file = next((p for fn, p in clip_index.items() if key in fn), None)
                if clip_file:
                    entries.append({
                        "start": start_ts,
                        "end": end_ts,
                        "text": text,
                        "clip": clip_file,
                    })

        return entries

    def _load_frames(self, clip_path: str) -> torch.Tensor:
        """Load clip as a (T, 1, H, W) grayscale float tensor, padded/truncated to max_frames."""
        cap = cv2.VideoCapture(clip_path)
        frames = []
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()

        n = len(frames)
        if n == 0:
            frames = [np.zeros(self.frame_size, dtype=np.uint8)]
            n = 1

        # Pad to max_frames
        pad = self.max_frames - n
        if pad > 0:
            frames += [np.zeros(self.frame_size, dtype=np.uint8)] * pad

        arr = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W)
        return torch.FloatTensor(arr).unsqueeze(1), n           # (T, 1, H, W), real_length

    def text_to_labels(self, text: str) -> torch.LongTensor:
        return torch.LongTensor([self.char2idx[c] for c in text if c in self.char2idx])

    def decode(self, indices: list[int]) -> str:
        return "".join(self.vocab[i] for i in indices if i != 0)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        frames, real_len = self._load_frames(entry["clip"])
        label = self.text_to_labels(entry["text"])
        return frames, label, real_len, entry["text"]


def collate_fn(batch):
    """Pad variable-length frames and concatenate labels for CTC."""
    frames_list, label_list, frame_lengths, texts = zip(*batch)

    # frames: (B, T, 1, H, W)
    frames = torch.stack(frames_list)
    # concatenate all labels into one tensor + lengths
    label_lengths = torch.LongTensor([len(l) for l in label_list])
    labels = torch.cat(label_list)
    frame_lengths = torch.LongTensor(frame_lengths)

    return frames, labels, label_lengths, frame_lengths, list(texts)
