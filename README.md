# 4471 Lip Reading

## Set up enviornment

Check up your nvidia driver version if >=570, you need a new driver to support wsl2.

Install uv to sync library: <br>
```curl -LsSf https://astral.sh/uv/install.sh | sh```

Restart Bash

Create venv with: ```uv venv```

Install Libraies (install from requirements.txt should be a lot easier)<br>
```uv pip install torch``` </br>
```uv pip install ultralytics``` </br>
```uv pip install opencv-python``` </br>
```uv pip install jupyter```</br>
```sudo apt install -y ffmpeg``` </br>
```uv pip install yt-dlp```</br>
or </br>
```sudo apt install -y ffmpeg``` </br>
```uv pip install -r requirements.txt``` </br>


Saving your current library set up </br>
```uv pip freeze > requirements.txt```

If you would like to test if GPU activated, paste into bash </br>
```python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('device count:', torch.cuda.device_count()); print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"```

## Download Media
```yt-dlp -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" -P "PASTE_YOUR_DEST_FOLDER" "PASTE_YOUTUBE_LINK_HERE"```

## Project structure

```
4471-Lip-Reading/
├── lip_extraction/        # core package
│   ├── detector.py        # YOLO face detection (auto GPU/CPU, auto-download weights)
│   ├── motion.py          # optical-flow lip motion filter
│   ├── pipeline.py        # orchestrates video → mouth clips
│   ├── dataset.py         # PyTorch Dataset (subtitle ↔ clip frames)
│   └── model.py           # LipNet: 3D-CNN + Transformer + CTC head
├── scripts/
│   ├── extract.py         # motion-based mouth clip extractor
│   ├── clean_subtitles.py # parse & reconstruct YouTube VTT into clean sentences
│   ├── crop_and_filter.py # subtitle-aligned crop + face consistency filter
│   └── train.py           # train the LipNet model
├── weights/               # YOLO model weights (gitignored)
├── checkpoints/           # training checkpoints (gitignored)
├── tests/
│   └── test_detector.py
├── youtube/               # downloaded media (gitignored)
│   ├── OpenU/
│   └── TVBNews/
├── extracted_clips/       # cropped clips (gitignored)
├── extracted_sub/         # subtitle outputs (gitignored)
└── README.md
```

---

## Scripts

### 1. Motion-based lip clip extractor

Scans a video for frames with face + lip motion and saves clips:

```bash
python3 -m scripts.extract \
    "youtube/OpenU/video_merged.mp4" \
    --output extracted_clips --clip-len 16
```

| Argument | Default | Description |
|----------|---------|-------------|
| `video` | — | Path to input video |
| `--output` | `clips` | Output directory |
| `--clip-len` | `16` | Frames per clip |

---

### 2. Subtitle cleaner

Parses a raw YouTube `.vtt` file (with inline `<c>` timing tags) and reconstructs
clean timestamped sentences. Output goes to `extracted_sub/`.

```bash
python3 -m scripts.clean_subtitles \
    "youtube/OpenU/video [id].yue-orig.vtt"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `vtt` | — | Path to `.vtt` file |
| `--gap` | `1.0` | Silence gap in seconds to split sentences |
| `--format` | `txt` | Output format: `txt` or `srt` |
| `--out-dir` | `extracted_sub` | Output directory |

Output format:
```
[00:00:24.519 --> 00:00:28.150]  你有冇諗過用啲乜
[00:00:28.720 --> 00:00:29.480]  達嘅需要咧？
```

---

### 3. Subtitle-aligned crop + face filter

Uses the clean subtitle timestamps to crop the video into per-sentence clips,
then checks each clip for consistent face visibility. Clips where the face
disappears or changes are deleted. Produces a `_filtered.txt` of kept entries.

```bash
python3 -m scripts.crop_and_filter \
    "youtube/OpenU/video_merged.mp4" \
    "extracted_sub/video_clean.txt"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `video` | — | Source video file |
| `subtitle` | — | Clean subtitle `.txt` file |
| `--out-dir` | `extracted_clips` | Output directory for clips |
| `--coverage` | `0.9` | Min fraction of frames requiring a visible face |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda:0` |

Outputs:
```
extracted_clips/<video_stem>/
    clip_0000_00h00m24s519.mp4
    clip_0002_00h00m28s720.mp4
    ...
extracted_sub/<subtitle_stem>_filtered.txt
```

---

## Recommended full pipeline

```bash
# 1. Download video
yt-dlp -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" \
    -P "youtube/OpenU" "YOUTUBE_URL"

# 2. Download subtitles
yt-dlp --write-auto-sub --sub-langs "yue-orig" --skip-download \
    -P "youtube/OpenU" "YOUTUBE_URL"

# 3. Clean subtitles
python3 -m scripts.clean_subtitles "youtube/OpenU/video [id].yue-orig.vtt"

# 4. Crop + filter clips
python3 -m scripts.crop_and_filter \
    "youtube/OpenU/video_merged.mp4" \
    "extracted_sub/video_clean.txt"

# 5. Train
python3 -m scripts.train \
    "extracted_sub/video_clean_filtered.txt" \
    "extracted_clips/video_merged/"
```

---

## 4. Train the lip reading model

Trains **LipNet** — a 3D-CNN + Transformer encoder + CTC head — on your subtitle-aligned clips.

```bash
python3 -m scripts.train \
    "extracted_sub/video_clean_filtered.txt" \
    "extracted_clips/video_merged/"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `subtitle` | — | `_filtered.txt` from crop_and_filter |
| `clips_dir` | — | Directory of `.mp4` clip files |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `4` | Batch size (reduce if VRAM limited) |
| `--lr` | `3e-4` | Peak learning rate |
| `--d-model` | `256` | Transformer hidden size |
| `--num-layers` | `4` | Transformer encoder depth |
| `--max-frames` | `100` | Max frames per clip (clips are padded/truncated) |
| `--checkpoint-dir` | `checkpoints` | Where to save weights |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |

Outputs:
```
checkpoints/
├── vocab.json              ← character vocabulary
├── best.pt                 ← best validation loss checkpoint
├── ckpt_epoch_001.pt
├── ckpt_epoch_002.pt
└── ...
```

### Model architecture

```
Input: (B, T, 1, 88, 88) grayscale frames
  │
  ▼
3D-CNN Frontend
  Conv3D 1→32 → BN → ReLU → MaxPool
  Conv3D 32→64 → BN → ReLU → MaxPool
  Conv3D 64→128 → BN → ReLU → AdaptiveAvgPool
  │  collapses spatial dims
  ▼
Linear projection → (B, T', 256)
  │
  ▼
Transformer Encoder (4 layers, 4 heads, Pre-LN)
  │
  ▼
Linear → log-softmax → (B, T', vocab_size)
  │
  CTC Loss (no forced alignment, blank = index 0)
```

### Notes on data size
- A single 45-min video typically yields ~300–1000 usable clips after filtering
- For good generalisation, aim for **5,000+ clips** across multiple speakers/videos
- Download more videos and run the full pipeline on each to grow the dataset
