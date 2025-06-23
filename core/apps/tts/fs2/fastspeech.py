import torch
from torch import nn

from ....models import EncoderBlock
from .adapter import VarianceAdapter
from .config import FastSpeechConfig


class FSEncoder(nn.Module):
    def __init__(self, config: FastSpeechConfig) -> None:
        super().__init__()
        self.transformer = nn.Sequential()
        for i in range(config.transformer_num_layer):
            self.transformer.add_module(
                "transformerBlock_" + str(i),
                EncoderBlock(
                    config.embed_size,
                    config.transformer_num_heads,
                    config.transformer_d_inner,
                    dropout=config.transformer_dropout,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class WaveformDecoder(nn.Module):
    def __init__(self, config: FastSpeechConfig) -> None:
        super().__init__()


class FastSpeech(nn.Module):
    def __init__(self, config: FastSpeechConfig) -> None:
        super().__init__()

        self.config = config
        self.encoder = FSEncoder(config)

        self.adapters = nn.Sequential()

        for i in range(config.adapter_num_layer):
            self.adapters.add_module("adapter_" + str(i), VarianceAdapter(config))

        self.decoder = FSEncoder(config)
        self.dense = nn.Linear(config.embed_size, config.n_mel_channels)

    def forward(
        self,
        x: torch.Tensor,
        d_control: float = 1.0,
        p_control: float = 1.0,
        e_control: float = 1.0,
    ):
        x = self.encoder(x)
        for adapter in self.adapters:
            x, mel_len, duration, pitch_pre, energy_pre = adapter(
                x, d_control, p_control, e_control
            )
        x = self.decoder(x)
        x = self.dense(x).transpose(1, 2)
        return (x, mel_len, duration, pitch_pre, energy_pre)
