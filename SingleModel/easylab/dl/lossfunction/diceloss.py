#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Dice loss function.

Dice loss is a kind of F1-score. Commonly used in segmentation task,
in which pixels classes are extremely imbalanced.

Reference: V-Net: Fully Convolutional Neural Networks for
Volumetric Medical Image Segmentation
Link: https://arxiv.org/pdf/1606.04797v1.pdf

Updating log:

V1.0 Default version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class dice loss implementation
    Version: V1.0
    """

    def __init__(self, smooth=1., reduce='mean', detail=False):
        super(DiceLoss, self).__init__()
        self.reduce = reduce
        self.smooth = smooth
        self.detail = detail
        # self.index = torch.tensor(select_index)
        # if select_index is not None else None

        return

    def forward(self, inputs, target):
        # inputs.float()

        N = target.size(0)  # Number of batch size
        C = inputs.size(1)  # Number of classes

        # One-hot encoding
        labels = target.unsqueeze(dim=1)
        one_hot = torch.zeros_like(inputs)
        target = one_hot.scatter_(1, labels.data, 1)

        input_ = F.softmax(inputs, dim=1)
        iflat = input_.contiguous().view(N, C, -1)
        tflat = target.contiguous().view(N, C, -1)
        intersection = (iflat * tflat).sum(dim=2)
        dice = (2. * intersection + self.smooth) / \
               (iflat.sum(dim=2) + tflat.sum(dim=2) + self.smooth)
        if self.detail:
            loss = (C * 1.0 - dice.sum(dim=1))
        elif self.reduce == 'mean':
            loss = (C * 1.0 - dice.sum(dim=1)).mean()
        elif self.reduce == 'sum':
            loss = N - dice.sum()

        return loss


if __name__ == '__main__':
    pass

