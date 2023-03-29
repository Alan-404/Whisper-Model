import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class MelExtractor(nn.Module):
    def __init__(self, mel_channels: int, d_model: int) -> None:
        super().__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=mel_channels, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)

        self.activation = F.gelu

        self.to(device)

    def forward(self, x: Tensor):
        x = self.conv1d_1(x)
        x = self.activation(x)
        x = self.conv1d_2(x)
        x = self.activation(x)

        return x