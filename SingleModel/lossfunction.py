
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class DiceLoss(nn.Module):
    def __init__(self, smooth=1., reduce='mean', detail=False):
        super(DiceLoss, self).__init__()
        self.reduce = reduce
        self.smooth = smooth
        self.detail = detail
        # self.index = torch.tensor(select_index) if select_index is not None else None

        return

    def forward(self, input, target):
        # input.float()

        N = target.size(0)
        C = input.size(1)
        labels = target.unsqueeze(dim=1)
        one_hot = torch.zeros_like(input)
        target = one_hot.scatter_(1, labels.data, 1)
        input_ = F.softmax(input, dim=1)
        iflat = input_.contiguous().view(N, C, -1)
        tflat = target.contiguous().view(N, C, -1)
        intersection = (iflat * tflat).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (iflat.sum(dim=2) + tflat.sum(dim=2) + self.smooth)
        if self.detail:
            loss = (C * 1.0 - dice.sum(dim=1))
        elif self.reduce == 'mean':
            loss = (C * 1.0 - dice.sum(dim=1)).mean()
        elif self.reduce == 'sum':
            loss = N - dice.sum()

        return loss


class Fscore(nn.Module):
    def __init__(self, smooth=1., beta=2, reduce='mean', detail=False):
        super(Fscore, self).__init__()
        self.reduce = reduce
        self.smooth = smooth
        self.detail = detail
        self.beta = beta ** 2
        # self.index = torch.tensor(select_index) if select_index is not None else None

        return

    def forward(self, input, target):
        # input.float()

        N = target.size(0)
        C = input.size(1)
        labels = target.unsqueeze(dim=1)
        one_hot = torch.zeros_like(input)
        target = one_hot.scatter_(1, labels.data, 1)
        input_ = F.softmax(input, dim=1)
        iflat = input_.contiguous().view(N, C, -1)
        tflat = target.contiguous().view(N, C, -1)
        intersection = (iflat * tflat).sum(dim=2)
        dice = ((1.0 + self.beta) * intersection + self.smooth) / (iflat.sum(dim=2) + self.beta * tflat.sum(dim=2) + self.smooth)
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
#         N = target.size(0)
#         C = pred.size(1)
#
#         label = torch.zeros_like(pred)
#         input_ = torch.sigmoid(pred)
#         label[:, 0] = (target != 0)
#         label[:, 1] = (((target == 4) + (target == 2)) != 0)
#         label[:, 2] = (target == 4)
#
#         iflat = input_.contiguous().view(N, C, -1)
#         tflat = label.contiguous().view(N, C, -1)
#         intersection = (iflat * tflat).sum(dim=2)
#         dice = (2. * intersection + self.smooth) / (iflat.sum(dim=2) + tflat.sum(dim=2) + self.smooth)
#         if self.detail:
#             loss = (C * 1.0 - dice.sum(dim=1))
#         elif self.reduce == 'mean':
#             loss = (C * 1.0 - die.sum(dim=1)).mean()
#         elif self.reduce == c'sum':
#             loss = N - dice.sum()

        # return loss


class RegionCrossEntropyLoss(nn.Module):
    def __init__(self, ):
        super(RegionCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='sum')

        return

    def forward(self, pred, target):
        pred_ = pred.max(1)[1]
        volume = ((target != 0) + (pred_ != 0)).float().sum()
        loss = self.ce(pred, target) / (target.size(0) * volume)

        return loss


class Brats19Loss(nn.Module):
    def __init__(self, sub='diceloss'):
        super(Brats19Loss, self).__init__()
        self.wt_dice = DiceLoss()
        self.tc_dice = DiceLoss()
        self.all_dice = DiceLoss()
        self.all_ce = RegionCrossEntropyLoss()
        self.sub = sub
        # self.index = torch.tensor(select_index) if select_index is not None else None

        return

    def forward(self, out, wt, tc, out_target, wt_target, tc_target):
        # input.float()

        wtd = self.wt_dice(wt, wt_target)
        tcd = self.tc_dice(tc, tc_target)
        if self.sub == 'diceloss':
            outd = self.all_dice(out, out_target)
        elif self.sub == 'ce':
            outd = self.all_ce(out, out_target)

        loss = wtd + tcd + outd

        return loss


class Brats19LossV2(nn.Module):
    def __init__(self, sub='diceloss', beta=2):
        super(Brats19LossV2, self).__init__()
        self.wt_dice = DiceLoss()
        self.tc_dice = DiceLoss()
        self.et_dice = DiceLoss()
        self.ed_dice = DiceLoss()
        self.ncr_dice = DiceLoss()
        self.wt_ce = Fscore(beta=beta)
        self.tc_ce = Fscore(beta=beta)
        self.et_ce = Fscore(beta=beta)
        self.ed_ce = Fscore(beta=beta)
        self.ncr_ce = Fscore(beta=beta)
        self.all_dice = DiceLoss()
        self.all_ce = Fscore(beta=beta)
        self.sub = sub
        # self.index = torch.tensor(select_index) if select_index is not None else None

        return

    def forward(self, out, wt, tc, et, ed, ncr, out_target, wt_target, tc_target, et_target, ed_target, ncr_target):
        # input.float()
        if self.sub == 'diceloss':
            wtd = self.wt_dice(wt, wt_target)
            tcd = self.tc_dice(tc, tc_target)
            etd = self.et_dice(et, et_target)
            edd = self.ed_dice(ed, ed_target)
            ncrd = self.ncr_dice(ncr, ncr_target)
            # outd = self.all_dice(out, out_target)

        elif self.sub == 'f2':
            wtd = self.wt_ce(wt, wt_target)
            tcd = self.tc_ce(tc, tc_target)
            etd = self.et_ce(et, et_target)
            edd = self.ed_ce(ed, ed_target)
            ncrd = self.ncr_ce(ncr, ncr_target)
            # outd = self.all_ce(out, out_target)

        outd = self.all_dice(out, out_target)
        loss = wtd + tcd + outd + etd + edd + ncrd

        return loss