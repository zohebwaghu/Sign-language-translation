"""
F1 -- Visual Encoder: ResNet-18 backbone + Landmark-Aware Spatial Attention.

Pipeline per frame:
  (3, 256, 256)
      -> ResNet-18 (frozen phase 1, unfrozen phase 2) -> (512, 8, 8) feature maps
      -> SpatialAttention (concat heatmap -> Conv -> Sigmoid -> element-wise ?)
      -> AdaptiveAvgPool2d(1)  -> (512,)
      -> Linear projection     -> (d_model,)

Batch processing: reshape (B, T, 3, H, W) -> (B*T, 3, H, W) -> backbone -> reshape back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import D_MODEL, IMG_SIZE, DROPOUT


class SpatialAttention(nn.Module):
    """
    Fuse CNN feature maps with a landmark heatmap via learned attention.

    The heatmap acts as a spatial prior -- it tells the model where hands
    and face are, so the network learns to weight those regions more.

    Architecture:
        [C+1, H, W]  (cat of CNN features + heatmap)
            -> Conv(kernel=1) -> BN -> ReLU
            -> Conv(kernel=1) -> Sigmoid
        = attention map [1, H, W]
        -> element-wise multiply with original CNN features
    """

    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, in_channels // 4, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(in_channels // 4)
        self.conv2 = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        heatmap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, C, H, W)  CNN feature maps.
            heatmap:  (N, H, W) or (N, 1, H, W) attention prior, or None.

        Returns:
            attended: (N, C, H, W)  attention-weighted features.
        """
        N, C, fH, fW = features.shape

        if heatmap is None:
            return features   # pass-through when no heatmap provided

        # Ensure heatmap has a channel dim and matches spatial size
        if heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(1)   # (N, 1, H, W)

        if heatmap.shape[-2:] != (fH, fW):
            heatmap = F.interpolate(
                heatmap, size=(fH, fW), mode="bilinear", align_corners=False
            )

        x = torch.cat([features, heatmap], dim=1)   # (N, C+1, fH, fW)
        x = F.relu(self.bn1(self.conv1(x)))
        attn = torch.sigmoid(self.conv2(x))          # (N, 1, fH, fW)

        return features * attn


class VisualEncoder(nn.Module):
    """
    Per-frame visual encoder.

    Args:
        d_model:   Output dimensionality (projected feature size).
        freeze_backbone: If True, ResNet weights are frozen (phase 1 training).
        dropout:   Dropout rate on the projected output.
    """

    # ResNet-18 outputs 512 feature channels at stride 32 for 256?256 input
    _BACKBONE_CHANNELS = 512

    def __init__(
        self,
        d_model: int = D_MODEL,
        freeze_backbone: bool = True,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # ?? Backbone: ResNet-18 without avgpool and FC ????????????????????
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Keep everything up to (but not including) the avgpool layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            self.freeze_backbone()

        # ?? Spatial Attention ????????????????????????????????????????????
        self.spatial_attention = SpatialAttention(self._BACKBONE_CHANNELS)

        # ?? Projection head ??????????????????????????????????????????????
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.dropout   = nn.Dropout(dropout)
        self.proj      = nn.Linear(self._BACKBONE_CHANNELS, d_model)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(
        self,
        frames: torch.Tensor,
        heatmaps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            frames:   (B, T, 3, H, W)  ImageNet-normalised video frames.
            heatmaps: (B, T, H, W) landmark heatmaps, or None.

        Returns:
            features: (B, T, d_model)  per-frame feature vectors.
        """
        B, T, C, H, W = frames.shape

        # Flatten batch ? time for parallel processing through backbone
        x = frames.view(B * T, C, H, W)         # (B*T, 3, H, W)
        feat = self.backbone(x)                  # (B*T, 512, fH, fW)

        # Flatten heatmaps similarly
        if heatmaps is not None:
            hm = heatmaps.view(B * T, H, W)     # (B*T, H, W)
        else:
            hm = None

        feat = self.spatial_attention(feat, hm)  # (B*T, 512, fH, fW)

        feat = self.pool(feat)                   # (B*T, 512, 1, 1)
        feat = feat.flatten(1)                   # (B*T, 512)
        feat = self.dropout(feat)
        feat = self.proj(feat)                   # (B*T, d_model)

        return feat.view(B, T, -1)              # (B, T, d_model)
