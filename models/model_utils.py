import torch
from torch import Tensor

def sandwich(x: Tensor):
    return x.reshape(x.size(0), -1, x.size(-1))

def pe_encode(sequence_length: int, embedding_size: int) -> Tensor:
    """Positional encoding as described in original attention is all you need paper"""

    pe = torch.zeros((sequence_length, embedding_size))
    pos = torch.arange(sequence_length).unsqueeze(1)
    pe[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )
    pe[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )

    return pe


def pe_encode_float(x: Tensor, max_freq: float, embedding_size: int) -> Tensor:
    pe = torch.zeros(list(x.shape) + [embedding_size], device=x.device)
    pos = (((x + 1) / 2) * max_freq).unsqueeze(-1)
    pe[..., 0::2] = torch.sin(
        pos
        / torch.pow(10000, torch.arange(0, embedding_size, 2, dtype=torch.float32, device=x.device) / embedding_size)
    )
    pe[..., 1::2] = torch.cos(
        pos
        / torch.pow(10000, torch.arange(1, embedding_size, 2, dtype=torch.float32, device=x.device) / embedding_size)
    )
    return pe