#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Common used loss function file

Some common used loss functions after test will be moved to here from individual
file.
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


# class Brats19Loss(nn.Module):
#     def __init__(self, smooth=1., reduce='mean', detail=False):
#         super(Brats19Loss, self).__init__()
#         self.reduce = reduce
#         self.smooth = smooth
#         self.detail = detail
#         # self.index = torch.tensor(select_index) if select_index is not None else None
#
#         return
#
#     def forward(self, pred, target):
#         # input.float()
#
#         labels = target.unsqueeze(dim=1)
#         one_hot = torch.zeros_like(pred)
#         target = one_hot.scatter_(1, labels.data, 1)
#
#         normal_dice = self.dice_compute(pred[:, 0], target[:, 0])
#         ET_dice = self.dice_compute(pred[:, 4], target[:, 4])
#         TC_dice = self.dice_compute(pred[:, 4] + pred[:, 2], target[:, 4] + target[:, 2])
#         WT_dice = self.dice_compute(pred[:, 1] + pred[:, 4] + pred[:, 2], target[:, 1] + target[:, 4] + target[:, 2])
#         ED_dice = self.dice_compute(pred[:, 2], target[:, 2])
#         NCR_dice = self.dice_compute(pred[:, 1], target[:, 1])
#         Background_dice = self.dice_compute(pred[:, 3], target[:, 3])
#
#         # stack = [normal_dice, ET_dice, TC_dice, WT_dice, ED_dice, NCR_dice, Background_dice]
#         stack = [normal_dice, ET_dice, ED_dice, NCR_dice, Background_dice]
#         count_dice = torch.stack(seq=stack, dim=1)
#         return (len(stack) - count_dice.sum(dim=1)).mean()
#
#     def dice_compute(self, pred, target, dims=(1, 2, 3)):
#         a = pred + target
#         overlap = (pred * target).sum(dim=dims) * 2
#         union = a.sum(dim=dims)
#         dice = (overlap + self.smooth) / (union + self.smooth)
#
#         return dice

class Brats19Loss(nn.Module):
    def __init__(self, smooth=1., reduce='mean', detail=False):
        super(Brats19Loss, self).__init__()
        self.reduce = reduce
        self.smooth = smooth
        self.detail = detail
        # self.index = torch.tensor(select_index) if select_index is not None else None

        return

    def forward(self, pred, target):
        # input.float()

        N = target.size(0)
        C = pred.size(1)

        label = torch.zeros_like(pred)
        input_ = torch.sigmoid(pred)
        label[:, 0] = (target != 0)
        label[:, 1] = (((target == 4) + (target == 2)) != 0)
        label[:, 2] = (target == 4)

        iflat = input_.contiguous().view(N, C, -1)
        tflat = label.contiguous().view(N, C, -1)
        intersection = (iflat * tflat).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (iflat.sum(dim=2) + tflat.sum(dim=2) + self.smooth)
        if self.detail:
            loss = (C * 1.0 - dice.sum(dim=1))
        elif self.reduce == 'mean':
            loss = (C * 1.0 - dice.sum(dim=1)).mean()
        elif self.reduce == 'sum':
            loss = N - dice.sum()

        return loss

