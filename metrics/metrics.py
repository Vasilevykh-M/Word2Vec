import torch

def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = torch.max(preds.data, 1)
    accuracy_ = (predicted == targets).sum().item() / targets.size(0)
    return accuracy_