import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def ib_loss(input_values, ib):
    """Computes the IB loss."""
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=100.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight

    def forward(self, input, target, features):
        probs = F.softmax(input, dim=1)
        one_hot = F.one_hot(target.long(), input.size(1)).float()
        grads = torch.sum(torch.abs(probs - one_hot), dim=1)
        features = features.view(grads.size(0), -1).mean(dim=1, keepdim=True)
        ib = grads * features
        ib = self.alpha / (ib + self.epsilon)
        ib = torch.clamp(ib, max=1000.0)
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        ce_loss = torch.clamp(ce_loss, min=0.0, max=10.0)
        final_loss = ib_loss(ce_loss, ib)
        return final_loss

def ib_focal_loss(input_values, ib, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean()

class IB_FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000., gamma=0.):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        num_classes = input.size(1)
        one_hot = F.one_hot(target.long(), num_classes).float()
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - one_hot), dim=1)
        ib = grads * features.view(-1)
        ib = self.alpha / (ib + self.epsilon)
        ce = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        return ib_focal_loss(ce, ib, self.gamma)

def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * 10
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        return focal_loss(ce, self.gamma)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :].to(x.device), index_float.T.to(x.device))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
