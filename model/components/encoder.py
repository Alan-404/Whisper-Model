import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.layer import EncoderLayer
from typing import Union, Callable
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, n: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.activation = F.gelu
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)])

        self.embedding_dim = embedding_dim

        self.to(device)

    def sinusoids(self, length, channels, max_timescale=10000) -> torch.Tensor:
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    
    def forward(self, x: torch.Tensor, length: int, padding_mask: torch.Tensor):
        x = x + self.sinusoids(length=length, channels=self.embedding_dim).to(device)

        for layer in self.layers:
            x = layer(x, padding_mask)

        return x