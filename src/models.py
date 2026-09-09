"""Classification head for CLIP-embedding multimodal inputs.

Design note: the CLIP encoders are frozen and used purely as feature
extractors, so the only trained component is this head. That choice was forced
by free-tier compute, but it means embeddings can be precomputed once and every
experiment then trains in seconds -- which is what made a 30-trial
hyperparameter search across three encoders affordable.

Input is the concatenation of L2-normalised CLIP image and text embeddings, so
`input_dim` is 2x the encoder's embedding width:

    ViT-B/32        512 -> 1024
    ViT-L/14@336px  768 -> 1536
    RN50x64        1024 -> 2048
"""

from typing import Sequence

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class MultimodalMLP(nn.Module):
    """MLP classification head over fused CLIP embeddings.

    The reported best configuration (ViT-L/14@336px) used
    hidden_dims=(512, 256, 128), dropout=0.4, activation='leakyrelu'.

    Args:
        input_dim: Width of the concatenated image+text embedding.
        hidden_dims: Widths of the hidden layers.
        num_classes: Output classes (2 for this task).
        dropout: Dropout probability applied after each hidden activation.
        activation: One of 'relu', 'leakyrelu', 'gelu'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        num_classes: int = 2,
        dropout: float = 0.4,
        activation: str = "relu",
    ):
        super().__init__()

        act = _ACTIVATIONS.get(activation.lower())
        if act is None:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}"
            )

        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), act(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fuse_embeddings(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """L2-normalise each modality, then concatenate.

    Normalising before concatenation prevents whichever modality happens to have
    the larger magnitude from dominating the fused representation.

    Limitation: concatenation is early fusion in its simplest form and cannot
    model interaction between the modalities -- which is where meme hostility
    actually lives. See docs/05_future_work.md.
    """
    image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=-1)
    text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)
    return torch.cat([image_emb, text_emb], dim=-1)
