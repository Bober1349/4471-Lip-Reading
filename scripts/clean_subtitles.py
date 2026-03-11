"""Parse a YouTube VTT subtitle file and reconstruct clean sentences."""

import re
import argparse
import os


def parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def strip_tags(text: str) -> str:
    """Remove all VTT/HTML tags, return plain text."""
    return re.sub(r"<[^>]+>", "", text).strip()


def parse_vtt_to_chars(path: str) -> list[tuple[float, str]]:
    """
    Extract (timestamp, text) pairs by reading inline <c> timing tags.
    Each character appears exactly once at its first spoken timestamp.
    Blocks that only repeat carryover text (no <c> tags) are skipped.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    chars = []
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line:
            continue

        block_start = parse_timestamp(ts_line.split("-->")[0].strip().split()[0])
        ts_idx = lines.index(ts_line)
        text_lines = lines[ts_idx + 1:]

        # Only process lines that contain <c> inline timing tags.
        # Plain-text carryover lines (repetitions) are intentionally ignored.
        active_line = next((l for l in reversed(text_lines) if "<c>" in l), None)
        if not active_line:
            continue

        # Split on VTT timestamp markers <HH:MM:SS.mmm>
        # Result: [base_text, ts1, tagged_text1, ts2, tagged_text2, ...]
        parts = re.split(r"<(\d{2}:\d{2}:\d{2}\.\d{3})>", active_line)

        # First segment: spoken at block_start time
        base = strip_tags(parts[0])
        if base:
            chars.append((block_start, base))

        # Remaining segments: each has an explicit timestamp
        i = 1
        while i + 1 < len(parts):
            ts = parse_timestamp(parts[i])
            text = strip_tags(parts[i + 1])
            if text:
                chars.append((ts, text))
            i += 2

    return chars


def reconstruct_sentences(
    chars: list[tuple[float, str]], gap_threshold: float = 1.0
) -> list[dict]:
    """
    Merge characters into sentences.
    A new sentence starts when:
      - The gap to the next character exceeds gap_threshold seconds
      - The current character ends with sentence-ending punctuation
    """
    sentence_enders = set("。？！…?!")
    sentences = []
    current_chars = []
    current_start = None

    for i, (ts, text) in enumerate(chars):
        if current_start is None:
            current_start = ts
        current_chars.append(text)

        gap = (chars[i + 1][0] - ts) if i + 1 < len(chars) else float("inf")
        ends_sentence = text and text[-1] in sentence_enders

        if ends_sentence or gap > gap_threshold:
            sentences.append({
                "start": current_start,
                "end": ts,
                "text": "".join(current_chars),
            })
            current_chars = []
            current_start = None

    return sentences


def write_srt(sentences: list[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n")
            f.write(f"{s['text']}\n\n")


def write_txt(sentences: list[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}]  {s['text']}\n")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct clean sentences from YouTube VTT subtitles.")
    parser.add_argument("vtt", help="Path to .vtt subtitle file")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: extracted_sub/<name>_clean.<format>)")
    parser.add_argument("--out-dir", default="extracted_sub", help="Output directory (default: extracted_sub)")
    parser.add_argument("--format", choices=["srt", "txt"], default="txt", help="Output format")
    parser.add_argument("--gap", type=float, default=1.0, help="Silence gap (seconds) to split sentences (default: 1.0)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.output:
        out_path = args.output
    else:
        basename = re.sub(r"\.vtt$", f"_clean.{args.format}", os.path.basename(args.vtt))
        out_path = os.path.join(args.out_dir, basename)

    print(f"📄 Parsing: {args.vtt}")
    chars = parse_vtt_to_chars(args.vtt)
    print(f"   {len(chars)} characters/tokens extracted")

    sentences = reconstruct_sentences(chars, gap_threshold=args.gap)
    print(f"   {len(sentences)} sentences reconstructed (gap threshold: {args.gap}s)")

    if args.format == "srt":
        write_srt(sentences, out_path)
    else:
        write_txt(sentences, out_path)

    print(f"✅ Saved to: {out_path}")


if __name__ == "__main__":
    main()

