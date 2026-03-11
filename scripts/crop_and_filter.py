"""
Pipeline:
  1. Read clean subtitle file, filter to Chinese-only entries (drop [音樂] etc.)
  2. Crop video segments per entry using ffmpeg
  3. Check each clip for consistent face+lip visibility throughout
  4. Delete clips that fail, write a filtered subtitle txt
"""

import re
import os
import argparse
import subprocess
import sys
import cv2
import sys
import os

# Allow running as `python3 -m scripts.crop_and_filter`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lip_extraction.detector import FaceDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_timestamp_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))


def strip_bracketed_tags(text: str) -> str:
    """Remove [音樂] style annotations."""
    return re.sub(r"\[.*?\]", "", text).strip()


# ---------------------------------------------------------------------------
# Step 1 – parse & filter subtitle
# ---------------------------------------------------------------------------

def load_and_filter(subtitle_path: str) -> list[dict]:
    """
    Parse clean subtitle txt.  Keep only entries that:
      - contain at least one Chinese character after stripping [tags]
      - have a minimum duration of 0.3s (skip single-char glitches)
    """
    pattern = re.compile(
        r"\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s+(.*)"
    )
    entries = []
    with open(subtitle_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            start_ts, end_ts, raw_text = m.group(1), m.group(2), m.group(3)
            clean_text = strip_bracketed_tags(raw_text)
            start = parse_timestamp_sec(start_ts)
            end = parse_timestamp_sec(end_ts)
            duration = end - start
            if not has_chinese(clean_text):
                continue
            if duration < 0.3:
                continue
            entries.append({
                "start": start,
                "end": end,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "text": clean_text,
            })
    return entries


# ---------------------------------------------------------------------------
# Step 2 – crop video with ffmpeg
# ---------------------------------------------------------------------------

def crop_clip(video_path: str, start: float, end: float, out_path: str) -> bool:
    """Crop [start, end] from video_path into out_path. Returns True on success."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "1",
        out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Step 3 – face consistency check
# ---------------------------------------------------------------------------

def check_face_consistency(
    clip_path: str,
    detector: FaceDetector,
    min_coverage: float = 0.9,
    max_center_drift: float = 0.25,
) -> bool:
    """
    Return True if:
      - A face is detected in >= min_coverage fraction of frames
      - The face centre doesn't drift more than max_center_drift * frame_width
        (ensures it's the same person throughout)
    """
    cap = cv2.VideoCapture(clip_path)
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    total = 0
    detected = 0
    cx_list, cy_list = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        faces = detector.detect(frame)
        if faces:
            detected += 1
            # Largest face
            face = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            cx_list.append((face[0] + face[2]) / 2)
            cy_list.append((face[1] + face[3]) / 2)
    cap.release()

    if total == 0:
        return False

    coverage = detected / total
    if coverage < min_coverage:
        return False

    # Check face centre stability across frames
    if cx_list and frame_w > 0:
        x_drift = (max(cx_list) - min(cx_list)) / frame_w
        y_drift = (max(cy_list) - min(cy_list)) / frame_h
        if x_drift > max_center_drift or y_drift > max_center_drift:
            return False

    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crop subtitle-aligned video clips and filter by face visibility."
    )
    parser.add_argument("video", help="Path to source video file")
    parser.add_argument("subtitle", help="Path to clean subtitle .txt file")
    parser.add_argument("--out-dir", default="extracted_clips", help="Directory for clips (default: extracted_clips)")
    parser.add_argument("--filtered-sub", default=None, help="Path for filtered subtitle txt (default: extracted_sub/<name>_filtered.txt)")
    parser.add_argument("--coverage", type=float, default=0.9, help="Min fraction of frames with a face (default: 0.9)")
    parser.add_argument("--device", default="auto", help="Device for face detection: auto|cpu|cuda:0")
    args = parser.parse_args()

    # Resolve output paths
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    clip_dir = os.path.join(args.out_dir, video_stem)
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs("extracted_sub", exist_ok=True)

    if args.filtered_sub:
        filtered_path = args.filtered_sub
    else:
        sub_stem = os.path.splitext(os.path.basename(args.subtitle))[0]
        filtered_path = os.path.join("extracted_sub", sub_stem + "_filtered.txt")

    # Step 1 – filter subtitles
    print(f"\n📄 Loading subtitles: {args.subtitle}")
    entries = load_and_filter(args.subtitle)
    print(f"   {len(entries)} entries after Chinese-only filter")

    # Step 2 & 3 – crop then check
    print(f"\n✂️  Cropping clips → {clip_dir}")
    print(f"🔍 Initialising face detector...")
    detector = FaceDetector(device=args.device)

    kept = []
    for i, entry in enumerate(entries):
        clip_name = (
            f"clip_{i:04d}_"
            f"{entry['start_ts'].replace(':', 'h', 1).replace(':', 'm', 1).replace('.', 's')}"
            f".mp4"
        )
        clip_path = os.path.join(clip_dir, clip_name)

        # Crop
        ok = crop_clip(args.video, entry["start"], entry["end"], clip_path)
        if not ok:
            print(f"  [{i+1}/{len(entries)}] ✗ ffmpeg failed – skipping: {clip_name}")
            continue

        # Face check
        consistent = check_face_consistency(clip_path, detector, min_coverage=args.coverage)
        duration = entry["end"] - entry["start"]
        if consistent:
            kept.append(entry)
            print(f"  [{i+1}/{len(entries)}] ✓ kept   ({duration:.1f}s) {entry['text'][:40]}")
        else:
            os.remove(clip_path)
            print(f"  [{i+1}/{len(entries)}] ✗ deleted ({duration:.1f}s) {entry['text'][:40]}")

    # Write filtered subtitle
    with open(filtered_path, "w", encoding="utf-8") as f:
        for e in kept:
            f.write(f"[{e['start_ts']} --> {e['end_ts']}]  {e['text']}\n")

    print(f"\n✅ Done — kept {len(kept)}/{len(entries)} clips")
    print(f"   Clips : {clip_dir}")
    print(f"   Filtered subtitle: {filtered_path}\n")


if __name__ == "__main__":
    main()
