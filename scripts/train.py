"""Train the LipNet model on subtitle-aligned mouth clips."""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lip_extraction.dataset import LipReadingDataset, collate_fn
from lip_extraction.model import LipNet


def compute_cer(pred_texts: list[str], true_texts: list[str]) -> float:
    """Character Error Rate (lower is better)."""
    total_chars, total_errors = 0, 0
    for pred, true in zip(pred_texts, true_texts):
        total_chars += len(true)
        # Simple edit distance (insertions + deletions + substitutions)
        d = [[0] * (len(pred) + 1) for _ in range(len(true) + 1)]
        for i in range(len(true) + 1):
            d[i][0] = i
        for j in range(len(pred) + 1):
            d[0][j] = j
        for i in range(1, len(true) + 1):
            for j in range(1, len(pred) + 1):
                cost = 0 if true[i - 1] == pred[j - 1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        total_errors += d[len(true)][len(pred)]
    return total_errors / max(total_chars, 1)


def run_epoch(model, loader, optimizer, ctc_loss, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_truths = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for frames, labels, label_lengths, frame_lengths, texts in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            log_probs = model(frames)              # (B, T', vocab)
            T_out = log_probs.size(1)
            # CTC requires input_lengths >= label_lengths
            input_lengths = torch.clamp(
                torch.full((frames.size(0),), T_out, dtype=torch.long),
                min=label_lengths
            )

            # CTCLoss expects (T', B, vocab)
            loss = ctc_loss(
                log_probs.permute(1, 0, 2),
                labels.to(device),
                input_lengths.to(device),
                label_lengths.to(device),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()

            # Greedy decode for CER
            preds = model.greedy_decode(frames)
            for p, t in zip(preds, texts):
                all_preds.append("".join(loader.dataset.dataset.vocab[i] for i in p
                                         if i < len(loader.dataset.dataset.vocab)))
                all_truths.append(t)

    cer = compute_cer(all_preds, all_truths)
    return total_loss / len(loader), cer


def main():
    parser = argparse.ArgumentParser(description="Train LipNet on filtered subtitle clips.")
    parser.add_argument("subtitle", help="Path to _filtered.txt")
    parser.add_argument("clips_dir", help="Directory containing .mp4 clip files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"🖥️  Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    print(f"📂 Loading dataset...")
    dataset = LipReadingDataset(
        args.subtitle, args.clips_dir, max_frames=args.max_frames
    )
    print(f"   {len(dataset)} samples | vocab size: {len(dataset.vocab)}")

    # Save vocab alongside checkpoints so inference can reuse it
    vocab_path = os.path.join(args.checkpoint_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(dataset.vocab, f, ensure_ascii=False, indent=2)
    print(f"   Vocab saved → {vocab_path}")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"   Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Model
    model = LipNet(
        vocab_size=len(dataset.vocab),
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 LipNet | {total_params/1e6:.1f}M parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"▶️  Resumed from epoch {ckpt['epoch']}")

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val CER':>9}")
    print("-" * 42)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, optimizer, ctc_loss, device, train=True)
        val_loss, val_cer = run_epoch(model, val_loader, optimizer, ctc_loss, device, train=False)
        scheduler.step()

        print(f"{epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {val_cer:>8.2%}")

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "vocab": dataset.vocab,
            "args": vars(args),
        }, ckpt_path)

        # Save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab": dataset.vocab,
                "args": vars(args),
            }, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"         ✅ New best model saved (val loss {best_val_loss:.4f})")

    print(f"\n✅ Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
