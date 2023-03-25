import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class ResidualConnection(nn.Module):
    def __init__(self, embedding_dim: int, dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)

        self.to(device)

    def forward(self, x: torch.Tensor, pre: torch.Tensor):
        x = self.dropout_layer(x)
        x = x + pre
        x = self.layer_norm(x)

        return x