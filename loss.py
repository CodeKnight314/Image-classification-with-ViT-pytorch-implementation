import torch 
import torch.nn as nn 

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




