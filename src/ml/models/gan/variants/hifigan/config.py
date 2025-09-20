from typing import List

from .....utils import load_yaml


class HiFiGANConfig:
    def __init__(self, config_path: str) -> None:
        self.config = load_yaml(config_path)

        # Audio
        self.sample_rate: int = self.config["audio"]["sample_rate"]
        self.n_fft: int = self.config["audio"]["n_fft"]
        self.n_mels: int = self.config["audio"]["n_mels"]
        self.hop_length: int = self.config["audio"]["hop_length"]

        # Generator
        self.gen_hidden_dim: int = self.config["generator"]["hidden_dim"]
        self.gen_in_channels: int = self.config["generator"]["in_channels"]
        self.gen_upsample_scales: List[int] = self.config["generator"]["upsampler"][
            "scales"
        ]
        self.gen_upsample_kernels: List[int] = self.config["generator"]["upsampler"][
            "kernels"
        ]
        self.gen_resblock_kernels: List[int] = self.config["generator"]["resblock"][
            "kernels"
        ]
        self.gen_resblock_dilations: List[List[int]] = self.config["generator"][
            "resblock"
        ]["dilations"]

        # Discriminator
        self.dis_hidden_dim: int = self.config["discriminator"]["hidden_dim"]

        self.dis_msd_num_blocks: int = self.config["discriminator"]["num_msd_blks"]
        self.dis_msd_kernels: List[int] = self.config["discriminator"]["msd_block"][
            "kernels"
        ]
        self.dis_msd_strides: List[int] = self.config["discriminator"]["msd_block"][
            "strides"
        ]
        self.dis_msd_groups: List[int] = self.config["discriminator"]["msd_block"][
            "groups"
        ]

        self.dis_mpd_periods: List[int] = self.config["discriminator"]["mpd_periods"]
        self.dis_mpd_num_blocks: int = len(self.dis_mpd_periods)
        self.dis_mpd_kernels: List[int] = self.config["discriminator"]["mpd_block"][
            "kernels"
        ]

        assert len(self.gen_upsample_scales) == len(self.gen_upsample_kernels), (
            "The number of upsample scales and kernels should be the same."
        )
        assert len(self.gen_resblock_kernels) == len(self.gen_resblock_dilations), (
            "The number of resblock kernels and dilations should be the same."
        )
