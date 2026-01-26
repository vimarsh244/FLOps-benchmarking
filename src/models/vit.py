"""Vision Transformer (ViT) model architectures."""

import torch
import torch.nn as nn
from typing import Optional

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ViT(nn.Module):
    """Vision Transformer wrapper using timm library.

    Supports various ViT variants from timm.
    """

    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        image_size: int = 224,
        in_channels: int = 3,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViT models. Install with: pip install timm")

        self.num_classes = num_classes
        self.image_size = image_size

        # create model from timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            img_size=image_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ViTSmall(nn.Module):
    """Small ViT for resource-constrained settings.

    A minimal ViT implementation without timm dependency.
    """

    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size

        num_patches = (image_size // patch_size) ** 2

        # patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # classification head (use cls token)
        x = x[:, 0]
        x = self.head(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # mlp with residual
        x = x + self.mlp(self.norm2(x))

        return x
