import nibabel as nib
import sys
from scipy.stats import mode
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool


def onehot(a):
    a = a.astype(np.int8)
    ncols = 5
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out

# comment = sys.argv[1]



def save_major(filename, source_dir):
    patient = nib.load('%s/%s.nii.gz' % (source_dir[0], filename))
    print(filename)
    preds = [onehot(nib.load('%s/%s.nii.gz' % (i, filename)).get_fdata())[:, :, :, :, np.newaxis] for i in source_dir]
    preds = np.concatenate(preds, axis=4)
    preds = preds.sum(axis=4)
    preds = np.argmax(preds, axis=3)
    new_data = nib.Nifti1Image(preds, patient.affine, patient.header)
    nib.save(new_data, os.path.join(save_dir, '%s.nii.gz' % filename))
    del new_data, preds, patient
    return


def save_prob(filename, source_dir):
    patient = nib.load('%s/%s.nii.gz' % (source_dir[0], filename))
    print(filename)
    preds = [nib.load('%s/%s.nii.gz' % (i, filename)).get_fdata()[:, :, :, np.newaxis] for i in source_dir]
    preds = np.concatenate(preds, axis=3)
    preds_new = np.zeros(shape=preds.shape[:3], dtype=np.uint8)
    for i in range(preds_new.shape[0]):
        for j in range(preds_new.shape[1]):
            for k in range(preds_new.shape[2]):
                preds_new[i][j][k] = np.random.choice(preds[i][j][k])
    new_data = nib.Nifti1Image(preds_new, patient.affine, patient.header)
    nib.save(new_data, os.path.join(save_dir, '%s.nii.gz' % filename))
    del new_data, preds_new, preds, patient

    return

if __name__ == '__main__':
    comment = 'test'
    save_dir = '%s_ensemble' % comment
    patients = pd.read_csv('statistic/survival_evaluation.csv')['BraTS19ID'].values

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    source_dir = [i for i in os.listdir() if 'single_test_v' in i]
    results = []
    pool = Pool()
    for filename in patients:
        results.append(pool.apply_async(func=save_major, args=(filename, source_dir)))

    pool.close()
    pool.join()

    [result.get() for result in results]