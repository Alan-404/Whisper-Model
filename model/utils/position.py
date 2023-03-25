import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self = self.to(device)

    def encode_length(self, length: int) -> torch.Tensor:
        pos = torch.arange(length)
        pos = pos.unsqueeze(-1)
        return pos.type(torch.float32).to(device)

    def encode_embedding(self) -> torch.Tensor:
        angles = torch.arange(self.embedding_dim)
        angles[1::2] = angles[0::2]
        angles = 1/(torch.pow(10000, angles/self.embedding_dim))

        angles = angles.unsqueeze(0)

        return angles.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.encode_length(x.size(1))
        angles = self.encode_embedding()

        angles_pos = torch.matmul(pos, angles)

        angles_pos[0::2] = torch.sin(angles_pos[0::2])
        angles_pos[1::2] = torch.cos(angles_pos[1::2])

        angles_pos = angles_pos.unsqueeze(0)

        x = x + angles_pos

        return x
