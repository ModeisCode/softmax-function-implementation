import torch

class Attention():
    def __init__(self) -> None:
        pass

    def softmax(self,xi: torch.Tensor):
        xjexps = 0.0
        for xj in xi:
            xjexps += torch.exp(xj)
        return torch.exp(xi) / xjexps
