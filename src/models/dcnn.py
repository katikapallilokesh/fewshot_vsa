"""
dcnn.py
-------
Deep Convolutional Neural Network for facial expression recognition.

Architecture mirrors the original paper exactly:
    Block 1+2 : Conv(64, 5×5) → BN → Conv(64, 5×5) → BN → MaxPool → Dropout(0.4)
    Block 3+4 : Conv(128, 3×3) → BN → Conv(128, 3×3) → BN → MaxPool → Dropout(0.4)
    Block 5+6 : Conv(256, 3×3) → BN → Conv(256, 3×3) → BN → MaxPool → Dropout(0.5)
    Head      : Flatten → Dense(128) → BN → Dropout(0.6) → Dense(num_classes)
    Activation: ELU throughout (avoids dying ReLU; suited to he_normal init)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two Conv-BN-ELU layers followed by MaxPool + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        padding = kernel_size // 2   # "same" padding

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DCNN(nn.Module):
    """
    Args:
        num_classes : number of output emotion classes
        in_channels : 1 for grayscale images (default), 3 for RGB
    """

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 64,  kernel_size=5, dropout=0.4),   # 48→24
            ConvBlock(64,          128, kernel_size=3, dropout=0.4),   # 24→12
            ConvBlock(128,         256, kernel_size=3, dropout=0.5),   # 12→6
        )

        # After 3× MaxPool2d(2): 48 → 6  →  256 × 6 × 6 = 9216
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He-normal initialisation for all Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x   # raw logits — loss function applies softmax internally

    def freeze_backbone(self, num_unfrozen_blocks: int = 1) -> None:
        """
        Freeze all feature-extraction layers except the last
        `num_unfrozen_blocks` ConvBlocks. Used when fine-tuning on few-shot data.
        """
        feature_children = list(self.features.children())
        freeze_up_to = len(feature_children) - num_unfrozen_blocks
        for i, block in enumerate(feature_children):
            for param in block.parameters():
                param.requires_grad = (i >= freeze_up_to)

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    @property
    def num_classes(self) -> int:
        return self.classifier[-1].out_features


def build_pretrain_model(num_classes: int) -> DCNN:
    """Fresh DCNN for the pretraining (transfer learning) stage."""
    return DCNN(num_classes=num_classes)


def build_fewshot_model(
    checkpoint_path: str,
    num_classes: int,
    num_unfrozen_blocks: int = 1,
) -> DCNN:
    """
    Loads the pretrained backbone, replaces the classification head with a
    new one for `num_classes` target emotions, and freezes all but the last
    `num_unfrozen_blocks` convolutional blocks.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pretrain_num_classes = checkpoint.get("num_classes",
                                           checkpoint["model_state"].get(
                                               "classifier.5.weight",
                                               torch.empty(1, 1)
                                           ).shape[0])

    # Build model with original head, load weights
    model = DCNN(num_classes=pretrain_num_classes)
    model.load_state_dict(checkpoint["model_state"])

    # Replace head for new number of classes
    model.classifier[-1] = nn.Linear(128, num_classes)
    nn.init.kaiming_normal_(model.classifier[-1].weight,
                             mode="fan_out", nonlinearity="relu")

    model.freeze_backbone(num_unfrozen_blocks=num_unfrozen_blocks)
    return model
