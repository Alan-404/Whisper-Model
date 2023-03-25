import torch
import torch.nn as nn
from model.utils.layer import DecoderLayer
from model.utils.position import PositionalEncoding
from typing import Union, Callable
from model.utils.mask import generate_mask

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, token_size: int, n: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token_size, embedding_dim=embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim=embedding_dim)
        self.layers = [DecoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)]
        self.to(device)
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, padding_mask: torch.Tensor, look_ahead_mask: torch.Tensor):
        
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, padding_mask, look_ahead_mask)
        return x