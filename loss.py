import torch 
import torch.nn as nn 

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        assert logits.shape == labels.shape and logits.size(1) != 1, "[ERROR] Logits and labels have incompatiable shapes or logits are shaped for binary classification."
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        return - (labels * log_probs).sum(dim=-1).mean().item()

class BCELoss(nn.Module): 
    def __init__(self): 
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor): 
        assert logits.shape == labels.shape and logits.size(1) == 1, "[ERROR] Logits and labels have incompatiable shapes or logits are shaped for multi-class classification."
        probs = torch.sigmoid(logits)
        probs_log = torch.log(probs + 1e-9)
        neg_probs_log = torch.log(1-probs + 1e-9)
        return - (labels * probs_log + (1-labels) * neg_probs_log).sum(dim=-1).mean().item()
    
