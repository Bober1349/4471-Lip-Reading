"""
LipNet model: 3D-CNN frontend + Transformer encoder + CTC head.

Architecture:
  3D-CNN  →  projects each frame to a feature vector
  Transformer encoder  →  models temporal dependencies across frames
  Linear + log-softmax  →  per-frame character probability distribution
  CTC loss during training (no forced alignment needed)
"""

import torch
import torch.nn as nn


class Frontend3D(nn.Module):
    """
    Spatio-temporal convolutional frontend.
    Input : (B, 1, T, H, W)  - grayscale frames
    Output: (B, T', feat_dim) - temporal feature sequence
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1 – capture fine-grained lip texture
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # Block 2
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # Block 3
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Collapse spatial dims → (B, 128, T', 1, 1)
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, H, W)
        out = self.net(x)              # (B, 128, T', 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # (B, 128, T')
        out = out.permute(0, 2, 1)        # (B, T', 128)
        return self.proj(out)              # (B, T', feat_dim)


class LipNet(nn.Module):
    """
    End-to-end lip reading model trained with CTC loss.

    Args:
        vocab_size  : number of output classes (including CTC blank at index 0)
        d_model     : transformer hidden size
        nhead       : number of attention heads
        num_layers  : transformer encoder depth
        dropout     : dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.frontend = Frontend3D(feat_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1, H, W) - padded frame sequences
        Returns:
            log_probs: (B, T', vocab_size)
        """
        # Rearrange to (B, C, T, H, W) for 3D conv
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)       # (B, 1, T, H, W)
        x = self.frontend(x)                # (B, T', d_model)
        x = self.transformer(x)             # (B, T', d_model)
        x = self.head(x)                    # (B, T', vocab_size)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def greedy_decode(self, x: torch.Tensor, blank_idx: int = 0) -> list[list[int]]:
        """
        Greedy CTC decode for inference.
        Returns a list of token index sequences (one per batch item).
        """
        log_probs = self.forward(x)         # (B, T', vocab)
        preds = log_probs.argmax(dim=-1)    # (B, T')
        results = []
        for seq in preds:
            # CTC collapse: remove consecutive duplicates then blanks
            collapsed = []
            prev = -1
            for idx in seq.tolist():
                if idx != prev:
                    if idx != blank_idx:
                        collapsed.append(idx)
                    prev = idx
            results.append(collapsed)
        return results
