import torch 
import torch.nn as nn 
import torch.optim as opt 
from typing import Tuple

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction = "mean"):
        super().__init__() 
        self.reduction = reduction
    
    def forward(self, logits : torch.Tensor, labels : torch.Tensor):
        assert logits.shape == labels.shape and logits.size(1) != 1, "[ERROR] Logits and labels have incompatiable shapes or logits are shaped for binary classification."
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        loss = - (labels * log_probs)
        return loss.sum(dim=-1).mean().item() if self.reduction == "mean" else loss.sum().item()

class BCELoss(nn.Module): 
    def __init__(self): 
        super().__init__()

    def forward(self, logits : torch.Tensor, labels : torch.Tensor): 
        assert logits.shape == labels.shape and logits.size(1) == 1, "[ERROR] Logits and labels have incompatiable shapes or logits are shaped for multi-class classification."
        probs = torch.sigmoid(logits)
        probs_log = torch.log(probs + 1e-9)
        neg_probs_log = torch.log(1-probs + 1e-9)
        return - (labels * probs_log + (1-labels) * neg_probs_log).sum(dim=-1).mean().item()

class HingeLoss(nn.Module): 
    def __init__(self): 
        super().__init__() 
    
    def forward(self, logits : torch.Tensor, labels : torch.Tensor): 
        assert logits.shape == labels.shape, "[ERROR] Logits and labels have incompatiable shapes."
        probs = torch.tanh(logits)
        loss = (1 - labels * probs).sum(dim=-1).mean().item()
        return max(0, loss)

def get_optimizer(model, lr : float, betas : Tuple[float], weight_decay : float): 
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
    return opt.Adam(model.parameters(), lr = lr, betas=betas, weight_decay=weight_decay)

def get_scheduler(optimizer : torch.optim, step_size : int, gamma : float): 
    """
    Helper function for defining learning rate scheduler -> may try to define my own for fun but who knows?

    Args: 
        optimizer (torch.optim): optimizer associated with the given learning rate scheduler 
        step_size (int): length of interval between each learning rate reduction 
        gamme (float): the rate at which the optimizer's learning rate decreases. New learning rate = lr * gamma at each step size interval
    """
    return opt.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)


