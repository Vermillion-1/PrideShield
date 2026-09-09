"""Loss functions for class-imbalanced classification.

The meme corpus is ~80.9% / 19.1%, so an untreated model can reach ~80.9%
accuracy by always predicting the majority class. Two strategies are provided;
which one is better was determined empirically per encoder by the Optuna study
(see notebooks/03_modeling/04_clip_mlp_optuna_final.ipynb), not assumed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017).

        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    The (1 - p_t)^gamma factor down-weights examples the model already
    classifies confidently, concentrating gradient on hard examples. With
    gamma=0 this reduces to standard cross-entropy.

    Args:
        alpha: Weight on the positive class.
        gamma: Focusing strength. Higher values suppress easy examples more.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)  # probability assigned to the true class
        loss = self.alpha * (1.0 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def inverse_frequency_weights(targets: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """Class weights inversely proportional to class frequency.

    The alternative imbalance strategy to FocalLoss. Pass the result as the
    `weight` argument of nn.CrossEntropyLoss.
    """
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights
