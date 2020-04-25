import pandas as pd
import numpy as np
import nibabel as nib
import os
import sys
import argparse
# from scipy.spatial.distance import directed_hausdorff
from add_metric import hd95

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--source', default='test0', type=str, help='source csv file')
parser.add_argument('--key', default='base', type=str, help='column name')

args = parser.parse_args()


def dice_coef(pred, target):
    # assert pred.shape == target.shape
    a = pred + target
    overlap = (pred * target).sum() * 2
    union = a.sum()
    epsilon = 0.0001
    dice = (overlap + epsilon) / (union + epsilon)

    return dice


def hausdorff_distance(pred, target):

    return hd95(pred, target, voxelspacing=None, connectivity=1)


def dice_brats19(pred, target):

    ET_dice = dice_coef((pred == 4).astype(np.int32), (target == 4).astype(np.int32))
    TC_dice = dice_coef((pred == 4).astype(np.int32) + (pred == 1).astype(np.int32), (target == 4).astype(np.int32) + (target == 1).astype(np.int32))
    WT_dice = dice_coef((pred != 0).astype(np.int32), (target != 0).astype(np.int32))
    ED_dice = dice_coef((pred == 2).astype(np.int32), (target == 2).astype(np.int32))
    NCR_dice = dice_coef((pred == 1).astype(np.int32), (target == 1).astype(np.int32))

    return ET_dice, TC_dice, WT_dice, ED_dice, NCR_dice


def recall(pred, target):
    if target.sum() == 0:
        return 1.0
    else:
        a = pred * target
        epsilon = 0.0001
        return a.sum() / (target.sum() + epsilon)


def evaluation_metrics(pred, target):

    et_pred = (pred == 4).astype(np.int32)
    et_target = (target == 4).astype(np.int32)
    tc_pred = (pred == 4).astype(np.int32) + (pred == 1).astype(np.int32)
    tc_target = (target == 4).astype(np.int32) + (target == 1).astype(np.int32)
    wt_pred = (pred != 0).astype(np.int32)
    wt_target = (target != 0).astype(np.int32)
    # dice value
    et_dice = dice_coef(et_pred, et_target)
    tc_dice = dice_coef(tc_pred, tc_target)
    wt_dice = dice_coef(wt_pred, wt_target)
    
    # sensitivity
    et_sensitivity = recall(et_pred, et_target)
    tc_sensitivity = recall(tc_pred, tc_target)
    wt_sensitivity = recall(wt_pred, wt_target)
    
    # specificity
    et_specificity = recall(np.ones_like(et_pred)-et_pred, np.ones_like(et_pred)-et_target)
    tc_specificity = recall(np.ones_like(et_pred)-tc_pred, np.ones_like(et_pred)-tc_target)
    wt_specificity = recall(np.ones_like(et_pred)-wt_pred, np.ones_like(et_pred)-wt_target)

    # hausdorff distance
    # try:
    #     et_hd = hausdorff_distance(et_pred, et_target)
    # except:
    #     et_hd = 1.0
    # try:
    #     tc_hd = hausdorff_distance(tc_pred, tc_target)
    # except:
    #     tc_hd = 1.0
    # try:
    #     wt_hd = hausdorff_distance(wt_pred, wt_target)
    # except:
    #     wt_hd = 1.0

    return et_dice, tc_dice, wt_dice, et_sensitivity, tc_sensitivity, wt_sensitivity, \
            et_specificity, tc_specificity, wt_specificity


def evaluation_simple(pred, target):
    et_pred = (pred == 4).astype(np.int32)
    et_target = (target == 4).astype(np.int32)
    tc_pred = (pred == 4).astype(np.int32) + (pred == 1).astype(np.int32)
    tc_target = (target == 4).astype(np.int32) + (target == 1).astype(np.int32)
    wt_pred = (pred != 0).astype(np.int32)
    wt_target = (target != 0).astype(np.int32)
    # dice value
    et_dice = dice_coef(et_pred, et_target)
    tc_dice = dice_coef(tc_pred, tc_target)
    wt_dice = dice_coef(wt_pred, wt_target)

    et_sum = et_pred.sum()
    tc_sum = tc_pred.sum()
    wt_sum = wt_pred.sum()
    # sensitivity
    # et_sensitivity = recall(et_pred, et_target)
    # tc_sensitivity = recall(tc_pred, tc_target)
    # wt_sensitivity = recall(wt_pred, wt_target)
    # 
    # # specificity
    # et_specificity = recall(np.ones_like(et_pred) - et_pred, np.ones_like(et_pred) - et_target)
    # tc_specificity = recall(np.ones_like(et_pred) - tc_pred, np.ones_like(et_pred) - tc_target)
    # wt_specificity = recall(np.ones_like(et_pred) - wt_pred, np.ones_like(et_pred) - wt_target)

    # hausdorff distance
    # try:
    #     et_hd = hausdorff_distance(et_pred, et_target)
    # except:
    #     et_hd = 1.0
    # try:
    #     tc_hd = hausdorff_distance(tc_pred, tc_target)
    # except:
    #     tc_hd = 1.0
    # try:
    #     wt_hd = hausdorff_distance(wt_pred, wt_target)
    # except:
    #     wt_hd = 1.0

    return et_dice, tc_dice, wt_dice, et_sum, tc_sum, wt_sum


def get_dict():
    em = {}
    em['dice'] = dice_coef
    em['iou'] = IOU
    em['recall'] = Recall
    em['precision'] = Precision

    return em

def load_nifti(name):
    prediction = '%s.nii.gz' % name
    truelabel = '%s_seg.nii.gz' % name
    target = nib.load(os.path.join('prediction', truelabel)).get_fdata()
    pred = nib.load(os.path.join('%s_prediction' % args.key, prediction)).get_fdata()

    return pred, target


if __name__ == '__main__':

    df = pd.read_csv('statistic/stats.csv')
    names = df['BraTS_2019_subject_ID'].values
    et_dice_ = []
    tc_dice_ = [] 
    wt_dice_ = [] 
    et_sensitivity_ = []
    tc_sensitivity_ = []
    wt_sensitivity_ = []
    et_specificity_ = []
    tc_specificity_ = []
    wt_specificity_ = []
    # et_hd_ = [] 
    # tc_hd_ = [] 
    # wt_hd_ = []
    et_sum_ = []
    tc_sum_ = []
    wt_sum_ = []
    
    for i in range(len(names)):
        print(i)
        pred, target = load_nifti(names[i])
        et_dice, tc_dice, wt_dice, et_sensitivity, tc_sensitivity, wt_sensitivity, \
        et_specificity, tc_specificity, wt_specificity = evaluation_metrics(pred, target)
        et_dice, tc_dice, wt_dice, et_sum, tc_sum, wt_sum = evaluation_simple(pred, target)

        et_dice_.append(et_dice)
        tc_dice_.append(tc_dice)
        wt_dice_.append(wt_dice)
        et_sensitivity_.append(et_sensitivity)
        tc_sensitivity_.append(tc_sensitivity)
        wt_sensitivity_.append(wt_sensitivity)
        et_specificity_.append(et_specificity)
        tc_specificity_.append(tc_specificity)
        wt_specificity_.append(wt_specificity)
        # et_hd_.append(et_hd)
        # tc_hd_.append(tc_hd)
        # wt_hd_.append(wt_hd)
        # et_sum_.append(et_sum)
        # tc_sum_.append(tc_sum)
        # wt_sum_.append(wt_sum)
        
    df['et_dice'] = et_dice_
    df['tc_dice'] = tc_dice_
    df['wt_dice'] = wt_dice_
    # df['et_hd'] = et_hd_
    # df['tc_hd'] = tc_hd_
    # df['wt_hd'] = wt_hd_
    df['et_sensitivity'] = et_sensitivity_
    df['tc_sensitivity'] = tc_sensitivity_
    df['wt_sensitivity'] = wt_sensitivity_
    df['et_specificity'] = et_specificity_
    df['tc_specificity'] = tc_specificity_
    df['wt_specificity'] = wt_specificity_
    # df['et_sum'] = et_sum_
    # df['tc_sum'] = tc_sum_
    # df['wt_sum'] = wt_sum_
    
    df = df[['Grade','BraTS_2019_subject_ID','NCR','ED','ET','WT','TC', \
             'et_dice', 'tc_dice', 'wt_dice', 'et_sensitivity', 'tc_sensitivity',\
             'wt_sensitivity', 'et_specificity', 'tc_specificity', 'wt_specificity',]]
    # df = df[['Grade','BraTS_2019_subject_ID','NCR','ED','ET','WT','TC', \
    #          'et_dice', 'tc_dice', 'wt_dice', 'et_sum', 'tc_sum',\
    #          'wt_sum',]]
    df.to_csv('statistic/%s.csv' % args.key, index=False)