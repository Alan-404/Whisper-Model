import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class MutliHeadAttention(nn.Module):
    def __init__(self, heads: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads

        self.head_samples = self.embedding_dim // self.heads

        self.linear_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.linear_output = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        
        self.to(device)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dk = torch.tensor(k.size(-1), dtype=torch.float32)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores/torch.sqrt(dk)

        if mask is not None:
            attention_scores += mask*(-1e15)

        attention_weights = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, v)

        return output
    
    def split(self, x: torch.Tensor):
        batch_size = x.size(0)
        length = x.size(1)

        x = x.reshape((batch_size, length, self.heads, self.head_samples))

        x_heads = x.permute((0, 2, 1, 3))

        return x_heads
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        batch_size = q.size(0)
        length = q.size(1) 

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        q_heads = self.split(qw)
        k_heads = self.split(kw)
        v_heads = self.split(vw)

        attention_output = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention_output = attention_output.permute((0, 2, 1, 3))
        attention_output = attention_output.reshape((batch_size, length, self.embedding_dim))

        output = self.linear_output(attention_output)

        return output
        