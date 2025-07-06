import math

import torch
from torch import nn

from .config import FastSpeechConfig
from .lr import LengthRegulator
from .predictors import VariancePredictor


class VarianceAdapter(nn.Module):
    def __init__(self, config: FastSpeechConfig):
        super().__init__()
        self.embed_size = config.embed_size
        self.hidden_size = config.adapter_hidden_size

        self.log_offset = config.adapter_log_offset

        self.duration_predictor = VariancePredictor(config)
        self.lr = LengthRegulator()
        self.pitch_predictor = VariancePredictor(config)
        self.energy_predictor = VariancePredictor(config)

        self.pitch_bins = nn.Parameter(
            torch.exp(
                torch.linspace(
                    math.log(config.adapter_f0_min),
                    math.log(config.adapter_f0_max),
                    config.adapter_nbins - 1,
                )
            ),
            requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(
                config.adapter_energy_min,
                config.adapter_energy_max,
                config.adapter_nbins - 1,
            ),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(self.hidden_size, self.embed_size)
        self.energy_embedding = nn.Embedding(self.hidden_size, self.embed_size)

    def forward(
        self,
        X: torch.Tensor,
        d_control: float = 1.0,
        p_control: float = 1.0,
        e_control: float = 1.0,
        max_length=None,
    ):
        log_duration = self.duration_predictor(X)
        duration = (
            torch.clamp(torch.round(torch.exp(log_duration)) - self.log_offset, min=0)
        ) * d_control
        X, mel_len = self.lr(X, duration, max_length=max_length)
        pitch_prediction = self.pitch_predictor(X) * p_control
        pitch_embedding = self.pitch_embedding(
            torch.bucketize(pitch_prediction, self.pitch_bins)
        )

        energy_prediction = self.energy_predictor(X) * e_control
        energy_embedding = self.energy_embedding(
            torch.bucketize(energy_prediction, self.energy_bins)
        )

        X = X + energy_embedding + pitch_embedding
        return X, mel_len, duration, pitch_prediction, energy_prediction
