import torch
from torch import nn

from .functional import align_melspec


class FastSpeechMelSpecLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, predictions, labels):
        (
            mel_predictions,
            _,
            duration_predictions,
            pitch_predictions,
            energy_predictions,
        ) = predictions

        (
            mel_labels,
            target_mel_len,
            duration_labels,
            pitch_labels,
            energy_labels,
        ) = labels

        log_duration_labels = torch.log(duration_labels.float() + 1)
        log_duration_labels.requires_grad = False
        pitch_labels.requires_grad = False
        energy_labels.requires_grad = False
        mel_labels.requires_grad = False

        mel_predictions = align_melspec(mel_predictions, mel_labels)

        mel_loss = self.mae_loss(mel_predictions, mel_labels)
        duration_loss = self.mse_loss(duration_predictions, duration_labels)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_labels)
        energy_loss = self.mse_loss(energy_predictions, energy_labels)

        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

        return total_loss, mel_loss, duration_loss, pitch_loss, energy_loss
