from __future__ import annotations

import copy

import torch
from torch import nn

try:
    from . import config
    from .grl import grad_reverse
except ImportError:
    import config
    from grl import grad_reverse


class EmotionMLP(nn.Module):
    def __init__(
        self,
        use_grl: bool = False,
        use_gender_grl: bool = False,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.use_grl = use_grl
        self.use_gender_grl = use_gender_grl
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.HIDDEN_DIM_1, config.HIDDEN_DIM_2),
            nn.ReLU(),
        )
        self.emotion_head = nn.Linear(config.HIDDEN_DIM_2, len(config.EMOTION_MAP))
        self.speaker_head = nn.Linear(config.HIDDEN_DIM_2, 24)
        self.gender_head = nn.Linear(config.HIDDEN_DIM_2, 2)

    def forward(
        self,
        x: torch.Tensor,
        lambda_grl: float = 1.0,
        lambda_speaker: float | None = None,
        lambda_gender: float | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.feature_extractor(x)
        outputs = {"z": z, "emotion_logits": self.emotion_head(z)}
        speaker_lambda = lambda_grl if lambda_speaker is None else lambda_speaker
        gender_lambda = lambda_grl if lambda_gender is None else lambda_gender
        if self.use_grl:
            outputs["speaker_logits"] = self.speaker_head(grad_reverse(z, speaker_lambda))
        if self.use_gender_grl:
            outputs["gender_logits"] = self.gender_head(grad_reverse(z, gender_lambda))
        return outputs

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)
