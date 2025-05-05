import torch
from torch import nn

from .functional import pad_listed_2Dtensors


class LengthRegulator(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        assert alpha > 0
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, duration, max_length=None):
        if not isinstance(duration, torch.Tensor):
            duration = torch.tensor(duration, dtype=torch.float32)

        duration = torch.round(duration.float() * self.alpha).long()

        b, len = duration.shape
        for i in range(b):
            if duration[i].sum() == 0:
                duration[i][torch.randint(0, len, (1,))] = 1

        output = []
        for phos, ds in zip(x, duration):
            X = []
            for pho, d in zip(phos, ds):
                if d != 0:
                    X.append(pho.repeat(int(d), 1))
            output.append(torch.cat(X, dim=0))

        output = pad_listed_2Dtensors(output, 0.0, max_length)

        return output, output.shape[1]
