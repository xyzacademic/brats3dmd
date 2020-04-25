#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""evaluation metrics.

Commonly used evaluation metrics after testing will be move to here from
individual file.
"""

import torch


def accuracy(predict, target):
    """
    Compute accuracy
    :param predict(torch.int64): Predictions. Shape=(N, label)
    :param target(torch.int64): True label. Shape=(N, class)
    :return (torch.float32): Accuracy
    """

    return predict.eq(target).sum().float() / target.size(0)


def balanced_accuracy(predict, target):
    c0 = target == 0
    c1 = target == 1
    c0_acc = predict[c0].eq(target[c0]).sum().float()/c0.sum().float()
    c1_acc = predict[c1].eq(target[c1]).sum().float()/c1.sum().float()

    return (c0_acc+c1_acc) / 2.0

def top5(predict, target):
    # maxk = max(topk)
    maxk = 5
    _, pred = predict.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []

    # for k in topk:
    k = 5
    correct_k = correct[:k].view(-1).float().sum()
    # res.append(correct_k)

    return correct_k / target.size(0)


def dice_coef(pred, target, dims=(1, 2, 3)):
    # assert pred.shape == target.shape
    target = target.float()
    pred = pred.float()
    a = pred + target
    overlap = (pred * target).sum(dim=dims) * 2
    union = a.sum(dim=dims)
    epsilon = 0.0001
    dice = overlap / (union + epsilon)

    return dice


def multi_dice_coef(pred, target, dims=(2, 3, 4)):
    epsilon = 0.0001
    N = target.size(0)
    C = pred.size(1)
    labels = target.unsqueeze(dim=1)
    one_hot = torch.zeros_like(pred)
    target = one_hot.scatter_(1, labels.data, 1)
    a = pred + target
    overlap = (pred * target).sum(dim=dims) * 2
    union = a.sum(dim=dims)
    epsilon = 0.0001
    dice = overlap / (union + epsilon)

    return dice


def dice_compute(pred, target, dims=(1, 2, 3)):
    epsilon = 0.0001
    a = pred + target
    overlap = (pred * target).sum(dim=dims) * 2
    union = a.sum(dim=dims)
    epsilon = 0.0001
    dice = overlap / (union + epsilon)

    return dice


def dice_brats19(pred, target_):
    labels = target_.unsqueeze(dim=1)
    one_hot = torch.zeros_like(pred)
    target = one_hot.scatter_(1, labels.data, 1)

    normal_dice = dice_compute(pred[:, 0], target[:, 0])
    ET_dice = dice_compute(pred[:, 4], target[:, 4])
    TC_dice = dice_compute(pred[:, 4]+pred[:, 1], target[:, 4]+target[:, 1])
    WT_dice = dice_compute(pred[:, 1]+pred[:, 4]+pred[:, 2], target[:, 1]+target[:, 4]+target[:, 2])
    ED_dice = dice_compute(pred[:, 2], target[:, 2])
    NCR_dice = dice_compute(pred[:, 1], target[:, 1])
    # Background_dice = dice_compute(pred[:, 3], target[:, 3])

    stack = [normal_dice, ET_dice, TC_dice, WT_dice, ED_dice, NCR_dice]
    count_dice = torch.stack(seq=stack, dim=1)
    return count_dice


# def dice_brats19(pred, target):
#     label = torch.zeros_like(pred)
#     label[:, 0] = (target != 0)
#     label[:, 1] = (((target == 4) + (target == 2)) != 0)
#     label[:, 2] = (target == 4)
#
#     WT_dice = dice_compute(pred[:, 0], label[:, 0])
#     TC_dice = dice_compute(pred[:, 1], label[:, 1])
#     ET_dice = dice_compute(pred[:, 2], label[:, 2])
#
#     stack = [WT_dice, TC_dice, ET_dice]
#     count_dice = torch.stack(seq=stack, dim=1)
#     return count_dice

def iou(pred, target, dims=(1, 2, 3)):
    # assert pred.shape == target.shape
    target = target.float()
    pred = pred.float()
    a = pred + target
    overlap = (pred * target).sum(dim=dims)
    union = (a > 0).sum(dim=dims).float()
    epsilon = 0.0001
    iou_value = overlap / (union + epsilon)

    return iou_value


def recall(pred, target, dims=(1, 2, 3)):
    target = target.float()
    pred = pred.float()
    a = pred * target
    epsilon = 0.0001
    return a.sum(dim=dims) / (target.sum(dim=dims) + epsilon)


def precision(pred, target, dims=(1, 2, 3)):
    target = target.float()
    pred = pred.float()
    a = pred * target
    epsilon = 0.0001
    return a.sum(dim=dims) / (pred.sum(dim=dims) + epsilon)


def get_dict():
    evaluation_method = {
        'accuracy': accuracy,
        'top5': top5,
        'dice': dice_coef,
        'iou': iou,
        'recall': recall,
        'precision': precision,
        'multi_dice_coef': multi_dice_coef,
        'dice_brats19': dice_brats19,
    }

    return evaluation_method

