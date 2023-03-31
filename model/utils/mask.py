import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def generate_padding_mask(tensor: torch.Tensor, pad_value: float)-> torch.Tensor:
    return torch.Tensor(tensor == pad_value).type(torch.int64)[:, np.newaxis, np.newaxis, :]

def generate_look_ahead_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.ones((length, length)), diagonal=1)

def generate_mask(tensor: torch.Tensor, pad_value: float):
    padding_mask = generate_padding_mask(tensor, pad_value).to(device)

    look_ahead_mask = generate_look_ahead_mask(tensor.size(1)).to(device)

    look_ahead_mask = torch.maximum(look_ahead_mask, padding_mask).to(device)

    return look_ahead_mask.to(device)