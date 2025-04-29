from typing import Dict

from ....utils import load_yaml


class FastSpeechConfig:
    def __init__(self, path: str):
        self.config: Dict = load_yaml(path)
        self.embed_size: int = self.config["embed_size"]
        self.n_mel_channels: int = self.config["n_mel_channels"]

        # Adapter
        self.adapter_num_layer: int = self.config["adapter"]["num_layer"]
        self.adapter_hidden_size: int = self.config["adapter"]["hidden_size"]
        self.adapter_filter_size: int = self.config["adapter"]["filter_size"]
        self.adapter_kernel_size: int = self.config["adapter"]["kernel_size"]
        self.adapter_dropout: float = self.config["adapter"]["dropout"]

        self.adapter_log_offset: float = self.config["adapter"]["log_offset"]
        self.adapter_f0_min: float = self.config["adapter"]["f0_min"]
        self.adapter_f0_max: float = self.config["adapter"]["f0_max"]
        self.adapter_nbins: int = self.config["adapter"]["n_bins"]
        self.adapter_energy_min: float = self.config["adapter"]["energy_min"]
        self.adapter_energy_max: float = self.config["adapter"]["energy_max"]

        # Transformer transformer
        self.transformer_num_layer: int = self.config["transformer"]["num_layer"]
        self.transformer_num_heads: int = self.config["transformer"]["num_heads"]
        self.transformer_d_inner: int = self.config["transformer"]["d_inner"]
        self.transformer_dropout: float = self.config["transformer"]["dropout"]
