from typing import OrderedDict

import librosa
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .config import HiFiGANConfig
from .parts import MPD, MSD, ResBlock


class HiFiGANGenerator(nn.Module):
    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        self.config = config

        module_list = []
        module_list.extend(
            [
                (
                    "conv_pre",
                    weight_norm(
                        nn.Conv1d(
                            config.gen_in_channels,
                            config.gen_hidden_dim,
                            kernel_size=7,
                            stride=1,
                            padding=3,
                        ),
                    ),
                ),
                ("leaky_relu_pre", nn.LeakyReLU(0.1)),
            ]
        )

        for i, (k, r) in enumerate(
            zip(config.gen_upsample_kernels, config.gen_upsample_scales)
        ):
            module_list.extend(
                [
                    (
                        f"upsample_{i}",
                        weight_norm(
                            nn.ConvTranspose1d(
                                config.gen_hidden_dim // (2**i),
                                config.gen_hidden_dim // (2 ** (i + 1)),
                                kernel_size=k,
                                stride=r,
                                padding=(k - r) // 2,
                            ),
                        ),
                    ),
                    (f"leaky_relu_{i}", nn.LeakyReLU(0.1)),
                    (
                        f"resblock_{i}",
                        ResBlock(
                            config.gen_hidden_dim // (2 ** (i + 1)),
                            config.gen_hidden_dim // (2 ** (i + 1)),
                            config.gen_resblock_kernels,
                            config.gen_resblock_dilations,
                        ),
                    ),
                ]
            )

        out_channels = config.gen_hidden_dim // (2 ** len(config.gen_upsample_scales))

        module_list.extend(
            [
                (
                    "conv_post",
                    weight_norm(
                        nn.Conv1d(
                            out_channels,
                            1,
                            kernel_size=7,
                            stride=1,
                            padding=3,
                        ),
                    ),
                ),
                ("tanh", nn.Tanh()),
            ]
        )

        self.seq = nn.Sequential(OrderedDict(module_list))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.seq(z)
        return z


class HiFiGANDiscriminator(nn.Module):
    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        self.config = config
        self.mpd = MPD(config)
        self.msd = MSD(config)


class HiFiGAN(nn.Module):
    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        self.config = config
        self.generator = HiFiGANGenerator(config)
        self.discriminator = HiFiGANDiscriminator(config)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def get_mel_spec(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        mel = librosa.feature.melspectrogram(
            y=x.numpy().astype("float32"),
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            hop_length=self.config.hop_length,
        )
        return torch.tensor(mel).unsqueeze(0)
