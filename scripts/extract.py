"""Command-line interface for lip-reading extraction pipeline."""

import argparse
from lip_extraction.pipeline import ExtractionPipeline
import os
import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract mouth clips from video.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Directory to save clips", default="clips")
    parser.add_argument("--clip-len", type=int, default=16, help="Number of frames per clip")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pipeline = ExtractionPipeline()
    clips = pipeline.process_video(args.video, clip_len=args.clip_len)
    for i, clip in enumerate(clips):
        out_path = os.path.join(args.output, f"clip_{i:04d}.mp4")
        height, width = clip[0].shape[:2]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height))
        for frame in clip:
            writer.write(frame)
        writer.release()
    print(f"saved {len(clips)} clips to {args.output}")


if __name__ == "__main__":
    main()
