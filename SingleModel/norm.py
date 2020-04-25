import nibabel as nib
import numpy as np
import os
from multiprocessing import Pool


# modality = ['t1', 't1ce', 't2', 'flair']


def normalize(dir, file):
    print(file)
    patient = nib.load(os.path.join(dir, file))
    img = patient.get_fdata()
    mean = img[img > 0].mean()
    std = img[img > 0].std()
    new_img = (img - mean) / std
    new_data = nib.Nifti1Image(new_img, patient.affine, patient.header)
    nib.save(new_data, os.path.join(target_path, file))

    return


if __name__ == '__main__':
    dir = '../source_data/MRI/train_data'
    target_path = '../source_data/MRI/test_data_norm'
    files = os.listdir(dir)

    p = Pool()
    for file in files:
        p.apply_async(normalize, args=(dir, file))

    p.close()
    p.join()