import torch 
import torch.nn as nn 
import torch.optim as opt 
from torch.nn import functional as F
from typing import Tuple

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the cross-entropy loss between logits and true labels.

        Args:
            logits (torch.Tensor): The logits output from a model (unnormalized scores).
            labels (torch.Tensor): The true labels, expected to be class indices.

        Returns:
            float: The calculated loss value, averaged over the batch.
        """
        probs = F.log_softmax(logits, dim=-1)
        loss = -(probs[range(logits.shape[0]), labels] + 1e-9)
        return loss.mean()

class BCELoss(nn.Module): 
    def __init__(self): 
        super().__init__()

    def forward(self, logits : torch.Tensor, labels : torch.Tensor): 
        """
        Calculates the binary cross-entropy loss between logits and true labels.

        Args:
            logits (torch.Tensor): The logits output from a model (unnormalized scores), expected to have shape [N, 1].
            labels (torch.Tensor): The true labels, expected to have the same shape as logits.

        Returns:
            float: The calculated loss value, averaged over all elements in the batch.
        """
        probs = torch.sigmoid(logits)
        probs_log = torch.log(probs + 1e-9)
        neg_probs_log = torch.log(1-probs + 1e-9)
        return - (labels * probs_log + (1-labels) * neg_probs_log).sum(dim=-1).mean()

class HingeLoss(nn.Module): 
    def __init__(self): 
        super().__init__() 
    
    def forward(self, logits : torch.Tensor, labels : torch.Tensor): 
        """
        Calculates the hinge loss for binary classification.

        Args:
            logits (torch.Tensor): The logits or scores output from a model.
            labels (torch.Tensor): The true labels, expected to have the same shape as logits and values of 1 or -1.

        Returns:
            float: The calculated loss value, averaged over all elements in the batch. The loss is constrained to be non-negative.
        """
        assert logits.shape == labels.shape, "[ERROR] Logits and labels have incompatiable shapes."
        probs = torch.tanh(logits)
        loss = (1 - labels * probs).sum(dim=-1).mean().item()
        return max(0, loss)

def get_optimizer(model, lr : float, momentum : Tuple[float], weight_decay : float): 
    """
    Helper function for defining optimizer 

    Args: 
        model : the model associated with the given optimizer 
        lr (float): learning rate for the optimizer 
        betas (Tuple[float]): a pair of floats
        weight_decay (float): determine rate of weight decay

    Returns:
        torch.optim : optimizer with the given parameters
    """
    return opt.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

def get_scheduler(optimizer: torch.optim.Optimizer, mode: str = 'min', factor: float = 0.1, patience: int = 10, verbose: bool = False, threshold: float = 0.02, threshold_mode: str = 'rel', cooldown: int = 0, min_lr: float = 1e-6, eps: float = 1e-8):
    """
    Helper function for defining a ReduceLROnPlateau learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): optimizer associated with the given learning rate scheduler.
        mode (str): One of 'min', 'max'. In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        verbose (bool): If True, prints a message to stdout for each update.
        threshold (float): Threshold for measuring the new optimum, to only focus on significant changes.
        threshold_mode (str): One of 'rel', 'abs'. In 'rel' mode, dynamic_threshold = best * ( 1 + threshold ) in 'max' mode or best * ( 1 - threshold ) in 'min' mode. In 'abs' mode, dynamic_threshold = best + threshold in 'max' mode or best - threshold in 'min' mode.
        cooldown (int): Number of epochs to wait before resuming normal operation after lr has been reduced.
        min_lr (float or list): A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively.
        eps (float): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored.

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: Initialized ReduceLROnPlateau scheduler.
    """
    return opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose, threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)