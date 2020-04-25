#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""Module name

Description

Reference:
Link:

Updating log:
V1.0 description
"""

import torch


def accuracy(predict, target):
    """
    Compute accuracy.
    :param predict(torch.int64): Predictions. Shape=(N, label)
    :param target(torch.int64): True label. Shape=(N, class)
    :return (torch.float32): Accuracy
    """

    return predict.eq(target).sum().float() / target.size(0)


def topk(predict, target, k=5):
    """
    Compute top K accuarcy.
    :param predict(torch.int64): Predictions. Shape=(N, label)
    :param target(torch.int64): True label. Shape=(N, class)
    :param k(int, tuple): K value.
    :return: (torch.float32): If K is a integer, return the Kth accuracy
                                else if K is a tuple, return a tuple accuracy
    """
    maxk = max(k)
    _, pred = predict.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    if isinstance(k, int):
        correct_k = correct[:k].view(-1).float().sum()

        return correct_k / target.size(0)

    elif isinstance(k, tuple):
        res = []
        for k_ in k:
            correct_k = correct[:k_].view(-1).float().sum()
            res.append(correct_k)

        return [correct_k / target.size(0) for correct_k in res]



if __name__ == '__main__':
    pass
