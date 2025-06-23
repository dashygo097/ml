from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def token2tensor(
    tokenizer, inputs, max_length: int = 512, padding: bool = True
) -> torch.Tensor:
    "turn huggingface tokenizered tokens into torch.Tensors (without other params)"
    outputs = []
    if padding:
        for i in inputs:
            encoded_i = tokenizer.encode(i, max_length=max_length, truncation=True)

            assert len(encoded_i) <= max_length, "error: OUT of index"
            while len(encoded_i) < max_length:
                encoded_i.append(0)
            outputs.append(encoded_i)
    else:
        for i in inputs:
            encoded_i = tokenizer.encode(i, max_length=max_length, truncation=True)
            outputs.append(encoded_i)
    return torch.tensor(outputs, dtype=torch.long)


def pad_listed_2Dtensors(
    x: List[torch.Tensor],
    value: float,
    max_length=None,
) -> torch.Tensor:
    "pho.shape = batch_size, Tmax, stft_length for pho in x"

    if max_length is None:
        max_len = max([pho.shape[0] for pho in x])
    else:
        max_len = max_length

    output = []

    for pho in x:
        padded_pho = F.pad(pho, (0, 0, 0, max_len - pho.shape[0]), "constant", value)
        output.append(padded_pho)

    output = torch.stack(output)
    # output.shape = listlen, max_len, stft_length
    return output


def align_melspec(feature: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    feature.shape = B, Tfmax, stft_length
    label.shape = B, Tlmax, stft_length

    return => (feature.shape -> label.shape)
    """

    aligned_feature = torch.zeros_like(label)

    if feature.shape[1] < label.shape[1]:
        aligned_feature = F.pad(
            feature, (0, 0, 0, label.shape[1] - feature.shape[1], 0, 0), "constant", 0.0
        )

    else:
        aligned_feature = feature[:, : label.shape[1], :]

    return aligned_feature


def align_one_hot(data: torch.Tensor, out_dim: int = -1) -> torch.Tensor:
    if data.dim() == 1:
        one_hot = torch.zeros(size=(data.shape[0], out_dim))
        for i in range(data.shape[0]):
            one_hot[i, data[i]] = 1

    elif data.dim() == 2:
        one_hot = torch.zeros(size=(data.shape[0], data.shape[1], out_dim))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                one_hot[i, j, data[i, j]] = 1

    else:
        raise ValueError("data.dim() should be 1 or 2")

    return one_hot
