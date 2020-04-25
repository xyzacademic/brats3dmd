from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms


class NiftiDataset(Dataset):
    '''
    Dataset designed for Nifti files
    '''
    def __init__(self, root, labels=True, nifti=False, transform=None, target_transform=None, ):
        '''

        :param root: Data path
        :param labels: Whether labels are there
        :param nifti: Whether return nifti files' name
        :param transform:
        :param target_transform:
        '''
        self.root = root
        self.nifti = nifti
        self.anat_list = [i for i in os.listdir(root) if 'anat' in i]
        self.anat_list.sort()
        self.lesion_list = [i for i in os.listdir(root) if 'lesion' in i]
        self.lesion_list.sort()
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.data = []

    def __getitem__(self, index):
        anat_file = self.anat_list[index]
        patient = nib.load(os.path.join(self.root, anat_file))
        if self.nifti:
            filename = anat_file
        # Load image data from patient. axis is C, Y, X, Z, dtype=float64
        img = np.expand_dims(a=patient.get_fdata(), axis=0).astype(np.float32)
        if self.labels:
            lesion_file = self.lesion_list[index]
            assert anat_file[:anat_file.find('.')] == lesion_file[
                                                      :lesion_file.find('.')]
            target = nib.load(os.path.join(self.root, lesion_file)).get_fdata().astype(np.int64)
        else:
            target = None

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.labels:
            return img, target
        else:
            if self.nifti:
                return img, filename
            else:
                return img

    def __len__(self):
        return len(self.anat_list)


class Brats19Dataset(Dataset):
    '''
    Dataset designed for Nifti files
    '''
    def __init__(self, root, patient_list=None, labels=True, nifti=False, transform=None, target_transform=None, ):
        '''

        :param root: Data path
        :param labels: Whether labels are there
        :param nifti: Whether return nifti files' name
        :param transform:
        :param target_transform:
        '''
        self.root = root
        self.nifti = nifti
        self.t1_list = [i + '_t1.nii.gz' for i in patient_list]
        self.t1ce_list = [i + '_t1ce.nii.gz' for i in patient_list]
        self.t2_list = [i + '_t2.nii.gz' for i in patient_list]
        self.flair_list = [i + '_flair.nii.gz' for i in patient_list]

        self.lesion_list = [i + '_seg.nii.gz' for i in patient_list]

        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.data = []

    def __getitem__(self, index):
        t1_file = self.t1_list[index]
        t1 = nib.load(os.path.join(self.root, t1_file))
        t1ce_file = self.t1ce_list[index]
        t1ce = nib.load(os.path.join(self.root, t1ce_file))
        t2_file = self.t2_list[index]
        t2 = nib.load(os.path.join(self.root, t2_file))
        flair_file = self.flair_list[index]
        flair = nib.load(os.path.join(self.root, flair_file))
        if self.nifti:
            filename = t1_file
        # Load image data from patient. axis is C, Y, X, Z, dtype=float64
        img = np.concatenate([np.expand_dims(a=t1.get_fdata(), axis=0).astype(np.float32),
                              np.expand_dims(a=t1ce.get_fdata(), axis=0).astype(np.float32),
                              np.expand_dims(a=t2.get_fdata(), axis=0).astype(np.float32),
                              np.expand_dims(a=flair.get_fdata(), axis=0).astype(np.float32),
                              ], axis=0)
        if self.labels:
            lesion_file = self.lesion_list[index]
            # assert anat_file[:10] == lesion_file[:10]
            target = nib.load(os.path.join(self.root, lesion_file)).get_fdata().astype(np.int64)
            # target[img.sum(axis=0) == 0] = 3
            # target_ = np.zeros_like(target)
            target = (target != 0).astype(np.int64)  # WT
            # target = (target == 4 ).astype(np.int64)  # ET
            # target = (target == 4).astype(np.int64) + (target == 2).astype(np.int64)  # TC
        else:
            target = None

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.labels:
            return img, target
        else:
            if self.nifti:
                return img, filename
            else:
                return img

    def __len__(self):
        return len(self.t1_list)


class Normalize(object):
    def __init__(self, view='all'):
        if view == 'XY':
            self.axes = (1, 2)
        elif view == 'ZX':
            self.axes = (2, 3)
        elif view == 'ZY':
            self.axes = (1, 3)
        elif view == 'all':
            self.axes = (1, 2, 3)
        else:
            raise TypeError

    def __call__(self, data):
        assert len(data.shape) == 4
        mean = data.mean(axis=self.axes, keepdims=True)
        std = data.std(axis=self.axes, keepdims=True)
        new_data = (data - mean) / (std + 0.000001)

        return new_data


class CropAndPad(object):
    def __init__(self, format_='Brats19', target='data'):
        if format_ == 'Brats19':
            self.target = target
            if self.target == 'data':
                self.shape = (4, 160, 192, 160)
            elif self.target == 'label':
                self.shape = (160, 192, 160)
            self.xstart = 38
            self.xend = 198  # 160 in 0' axis
            self.ystart = 28
            self.yend = 220  # 192 in 1' axis
        else:
            raise AttributeError

    def __call__(self, data):
        if self.target == 'data':
            new_data = np.zeros(shape=self.shape, dtype=np.float32)
            new_data[:, :, :, 5:] = data[:, self.xstart:self.xend, self.ystart:self.yend, :]
        elif self.target == 'label':
            new_data = np.zeros(shape=self.shape, dtype=np.int64)
            new_data[:, :, 5:] = data[self.xstart:self.xend, self.ystart:self.yend, :]
        return new_data


class RandomFlip(object):
    def __init__(self, view='Y', prob=0.5):
        self.prob = prob
        if view == 'Y':
            self.axes = 1

    def __call__(self, data):
        assert len(data.shape) == 4
        if np.random.randn() > self.prob:
            data = np.flip(data, axis=self.axes)

        return data


if __name__ == '__main__':

    path = '../source_data/ATLAS_data'
    transform = transforms.Compose(
        [
            Normalize(view='XY')
        ]
    )
    dataset = NiftiDataset(root=path, labels=False, nifti=True, transform=transform)

    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    p = iter(dataloader)
    data, filename= p.next()
