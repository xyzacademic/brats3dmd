import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool


plt.switch_backend('agg')


def save_image(patient, image_dir, seg_dir):
    print(patient)
    t1ce = nib.load('%s/%s_t1ce.nii.gz' % (image_dir, patient)).get_fdata()
    t1 = nib.load('%s/%s_t1.nii.gz' % (image_dir, patient)).get_fdata()
    flair = nib.load('%s/%s_flair.nii.gz' % (image_dir, patient)).get_fdata()
    t2 = nib.load('%s/%s_t2.nii.gz' % (image_dir, patient)).get_fdata()
    label = nib.load('%s/%s.nii.gz' % (seg_dir, patient)).get_fdata().astype(np.uint8)
    wt_label = np.zeros_like(label, dtype=np.float32)
    wt_label[label != 0] = 1
    color_label = np.zeros((240, 240, 155, 3), dtype=np.uint8)
    color_label[label == 4, :] = [255, 0, 0]  # red
    color_label[label == 1, :] = [255, 255, 0]
    color_label[label == 2, :] = [0, 0, 255]
    y_idx = wt_label.sum(axis=(1, 2)).argmax()
    x_idx = wt_label.sum(axis=(0, 2)).argmax()
    z_idx = wt_label.sum(axis=(0, 1)).argmax()

    plt.figure(figsize=(10, 6), dpi=200)
    plt.subplot(3, 5, 1)
    plt.title('T1')
    plt.imshow(t1[:, :, z_idx])
    plt.subplot(3, 5, 2)
    plt.title('T1ce')
    plt.imshow(t1ce[:, :, z_idx])
    plt.subplot(3, 5, 3)
    plt.title('T2')
    plt.imshow(t2[:, :, z_idx])
    plt.subplot(3, 5, 4)
    plt.title('Flair')
    plt.imshow(flair[:, :, z_idx])
    plt.subplot(3, 5, 6)
    plt.title('T1')
    plt.imshow(t1[:, x_idx, :])
    plt.subplot(3, 5, 7)
    plt.title('T1ce')
    plt.imshow(t1ce[:, x_idx, :])
    plt.subplot(3, 5, 8)
    plt.title('T2')
    plt.imshow(t2[:, x_idx, :])
    plt.subplot(3, 5, 9)
    plt.title('Flair')
    plt.imshow(flair[:, x_idx, :])
    plt.subplot(3, 5, 11)
    plt.title('T1')
    plt.imshow(t1[y_idx, :, :])
    plt.subplot(3, 5, 12)
    plt.title('T1ce')
    plt.imshow(t1ce[y_idx, :, :])
    plt.subplot(3, 5, 13)
    plt.title('T2')
    plt.imshow(t2[y_idx, :, :])
    plt.subplot(3, 5, 14)
    plt.title('Flair')
    plt.imshow(flair[y_idx, :, :])
    plt.subplot(3, 5, 5)
    plt.title('XY')
    plt.imshow(color_label[:, :, z_idx])
    plt.subplot(3, 5, 10)
    plt.title('ZY')
    plt.imshow(color_label[:, x_idx, :])
    plt.subplot(3, 5, 15)
    plt.title('ZX')
    plt.imshow(color_label[y_idx, :, :])
    plt.show()
    plt.savefig('%s/%s.jpg' % (save_dir, patient))
    plt.close()

    del t1, t1ce, t2, flair, label, wt_label, color_label
    return

if __name__ == '__main__':
    file = 'survival_evaluation'
    save_dir = 'pred_test_image'
    image_dir = '../source_data/test_data_cp'
    seg_dir = 'test_ensemble'
    patients = pd.read_csv('statistic/%s.csv' % file)['BraTS19ID'].values

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = []
    pool = Pool()
    for patient in patients:
        results.append(pool.apply_async(save_image, args=(patient, image_dir, seg_dir)))

    pool.close()
    pool.join()
    [result.get() for result in results]