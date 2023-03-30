import torch
import torch.nn as nn
from typing import Union, Callable
from .attention import MutliHeadAttention
from .mlp import MLP
from .residual import ResidualConnection

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.multi_head_attention = MutliHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.mlp = MLP(d_ff=d_ff, embedding_dim=embedding_dim, activation=activation)

        self.residual_connection_1 = ResidualConnection(embedding_dim=embedding_dim, dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_2 = ResidualConnection(embedding_dim=embedding_dim, dropout_rate=dropout_rate, eps=eps)

        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # sublayer 1
        q = k = v = x
        attention_output = self.multi_head_attention(q, k, v, mask)
        sub_layer_1 = self.residual_connection_1(attention_output, x)

        # sublayer 2
        mlp_output = self.mlp(sub_layer_1)
        sub_layer_2 = self.residual_connection_2(mlp_output, sub_layer_1)

        return sub_layer_2
    
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.attention = MutliHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.cross_attention = MutliHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.mlp = MLP(d_ff=d_ff ,embedding_dim=embedding_dim, activation=activation)

        self.residual_connection_1 = ResidualConnection(embedding_dim=embedding_dim, dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_2 = ResidualConnection(embedding_dim=embedding_dim, dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_3 = ResidualConnection(embedding_dim=embedding_dim, dropout_rate=dropout_rate, eps=eps)

        self.to(device)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, padding_mask: torch.Tensor, look_ahead_mask: torch.Tensor):
        # sublayer 1
        q = k = v = x
        attention_output = self.attention(q, k, v, look_ahead_mask)
        sub_layer_1 = self.residual_connection_1(attention_output, x)

        # sublayer 2
        q = sub_layer_1
        k = v = encoder_output
        cross_output = self.cross_attention(q, k, v, padding_mask)
        sub_layer_2 = self.residual_connection_2(cross_output, sub_layer_1)

        # sublayer 3
        mlp_input = sub_layer_2
        mlp_output = self.mlp(mlp_input)
        sub_layer_3 = self.residual_connection_3(mlp_output, sub_layer_2)

        return sub_layer_3