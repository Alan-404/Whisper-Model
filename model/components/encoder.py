import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.layer import EncoderLayer
from typing import Union, Callable
import numpy as np

from model.utils.mel import MelExtractor

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, n: int, n_mels: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.mel_extractor = MelExtractor(mel_channels=n_mels, d_model=embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)])

        self.embedding_dim = embedding_dim

        self.to(device)

    def sinusoids(self, length, channels, max_timescale=10000) -> torch.Tensor:
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """ 
            x: shape = (batch_size, n_mels, time)
        """
        x = self.mel_extractor(x) # (batch_size, d_model, time)

        x = x.transpose(-1, -2) # (batch_size, time, d_model)
        x = x + self.sinusoids(length=x.size(1), channels=self.embedding_dim).to(device)
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x