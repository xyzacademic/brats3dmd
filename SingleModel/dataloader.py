from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage import rotate, shift



class Brats19Dataset(Dataset):
    '''
    Dataset designed for Nifti files
    '''
    def __init__(self, root, patient_list=None, labels=True, nifti=False, data_keyword=[], target_keyword=[],
                 data_transform=None, target_transform=None, both_transform=None ):
        '''

        :param root: Data path
        :param labels: Whether labels are there
        :param nifti: Whether return nifti files' name
        :param transform:
        :param target_transform:
        '''
        self.root = root
        self.nifti = nifti
        self.data_list = [[i + '_%s.nii.gz' % keyword for i in patient_list] for keyword in data_keyword]
        self.lesion_list = [[i + '_%s.nii.gz' % keyword for i in patient_list] for keyword in target_keyword]

        self.labels = labels
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.both_transform = both_transform
        self.data = []

    def __getitem__(self, index):

        single_data = [nib.load(os.path.join(self.root, self.data_list[i][index])).get_fdata().astype(np.float32)[np.newaxis, :] for i in range(len(self.data_list))]
        if self.nifti:
            filename = self.data_list[0][index]
        # Load image data from patient. axis is C, Y, X, Z, dtype=float32
        img = np.concatenate(single_data, axis=0)
        if self.labels:
            lesion_data = [nib.load(os.path.join(self.root, self.lesion_list[i][index])).get_fdata().astype(np.int64)[np.newaxis, :] for i in range(len(self.lesion_list))]
            # assert anat_file[:10] == lesion_file[:10]
            target = np.concatenate(lesion_data, axis=0)
            # target[img.sum(axis=0) == 0] = 3
            # target_ = np.zeros_like(target)
            # target = (target != 0).astype(np.int64)  # WT
            # target = (target == 4 ).astype(np.int64)  # ET
            # target = (target == 4).astype(np.int64) + (target == 1).astype(np.int64)  # TC
            # target = (target == 1).astype(np.int64)  # NCR/NEC
            # target = (target == 2).astype(np.int64)  # ED
        else:
            target = None

        if self.data_transform is not None:
            img = self.data_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.both_transform is not None:
            img, target = self.both_transform([img, target])

        if self.labels:
            return img, target
        else:
            if self.nifti:
                return img, filename
            else:
                return img

    def __len__(self):
        return len(self.data_list[0])


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


class RandomIntensity(object):
    def __init__(self, prob=0.5, shift=0.1):
        self.prob = prob
        self.shift = shift

    def __call__(self, data):
        if np.random.rand() > self.prob:
            out = np.random.uniform(1-self.shift, 1+self.shift)
            data = data + out

        return data


class PartialRandomIntensity(object):
    def __init__(self, prob=0.5, shift=0.1):
        self.prob = prob
        self.shift = shift

    def __call__(self, data):
        img = data[0]
        et = data[1][3]
        ed = data[1][4]
        ncr = data[1][5]

        # if np.random.uniform(0, 1) > 0.5:
        #     out = np.random.uniform(1-self.shift, 1+self.shift)
        #     img[:, et == 1] *= out
        # if np.random.uniform(0, 1) > 0.5:
        #     out = np.random.uniform(1 - self.shift, 1 + self.shift)
        #     img[:, ed == 1] *= out
        # if np.random.uniform(0, 1) > 0.5:
        #     out = np.random.uniform(1-self.shift, 1+self.shift)
        #     img[:, ncr == 1] *= out
        if np.random.uniform(0, 1) > self.prob:
            out = np.random.uniform(1, 1+self.shift)
            temp = img[[0, 2]]
            temp[:, et == 1] += out
            img[[0, 2]] = temp
            # img[[0, 2], et == 1] *= out
        if np.random.uniform(0, 1) > self.prob:
            out = np.random.uniform(1, 1 + self.shift)
            temp = img[[1, 3]]
            temp[:, ed == 1] += out
            img[[1, 3]] = temp
            # img[[1, 3], ed == 1] *= out
        if np.random.uniform(0, 1) > self.prob:
            out = np.random.uniform(1-self.shift, 1)
            temp = img[[0, 2]]
            temp[:, ncr == 1] += out
            img[[0, 2]] = temp
            # img[[0, 2], ncr == 1] *= out
        return img, data[1]


class FlipAug(object):
    def __init__(self, prob=0.5, mode='single'):
        self.prob = prob
        self.mode = mode

    def __call__(self, data):
        if self.mode == 'single':
            new_data = np.concatenate([data, data[:, ::-1, :, :]], axis=0)
            return new_data
        elif self.mode == 'dual':
            img = data[0]
            new_data = np.concatenate([img, img[:, ::-1, :, :]], axis=0)
            return new_data, data[1]


class RandomRotate(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if np.random.randn() > 0.5:
            axis = np.random.choice([1, 2, 3], 2, replace=False)
            degree = np.random.choice(np.arange(3), 1)
            degree = np.random.choice([-1, 1]) * degree
            img = rotate(data[0], angle=degree, axes=axis, reshape=True).astype(np.float32)
            label = rotate(data[1], angle=degree, axes=axis, reshape=True).astype(np.int64)

            return img, label
        else:
            return data[0], data[1]


class CropAndPad(object):
    def __init__(self, format_='Brats19', target='data', channel=1):
        if format_ == 'Brats19':
            self.target = target
            if self.target == 'data':
                self.shape = (channel, 160, 192, 160)
            elif self.target == 'label':
                self.shape = (channel, 160, 192, 160)

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
            new_data[:, :, :, 5:] = data[:, self.xstart:self.xend, self.ystart:self.yend, :]
        return new_data

class RandomCrop(object):
    def __init__(self, format_='Brats19', shape=(160, 192, 128)):
        if format_ == 'Brats19':
            self.shape = shape
            self.size = (240, 240, 155)

    def __call__(self, data):
        x = np.random.choice(np.arange(0, self.size[0] - self.shape[0]))
        y = np.random.choice(np.arange(0, self.size[1] - self.shape[1]))
        z = np.random.choice(np.arange(0, self.size[2] - self.shape[2]))

        img = data[0][:, x:x+self.shape[0], y:y+self.shape[1], z:z+self.shape[2]]
        label = data[1][:, x:x+self.shape[0], y:y+self.shape[1], z:z+self.shape[2]]

        return img, label

class RandomFlip(object):
    def __init__(self, view=None, prob=0.5):
        self.prob = prob
        if view == 'Y':
            self.axes = 1
        elif view == 'X':
            self.axes = 2
        elif view == 'Z':
            self.axes = 3
        else:
            self.axes = None

    def __call__(self, data):
        # assert len(data.shape) == 4
        if np.random.randn() > self.prob:
            if self.axes is None:
                axes = np.random.choice(np.arange(1, 4))
                # data = np.flip(data[0], axis=axes)
                # label = np.flip(data[1], axis=axes - 1)
            else:
                data = np.flip(data[0], axis=self.axes)
                label = np.flip(data[1], axis=self.axes-1)

            return data, label
        else:
            return data


if __name__ == '__main__':
    import pandas as pd

    data_path = '../source_data/train_data'
    data_keyword = ['flair']
    target_keyword = ['seg', 'wt', 'tc']
    data_transform = transforms.Compose(
        [
            CropAndPad('Brats19', target='data', channel=len(data_keyword)),
            Normalize(view='all')
        ]
    )

    label_transform = transforms.Compose(
        [
            CropAndPad('Brats19', target='label', channel=len(target_keyword)),
        ]
    )

    train_list = pd.read_csv('statistic/train_40.csv')['BraTS_2019_subject_ID'].values
    trainset = Brats19Dataset(root=data_path, patient_list=train_list, labels=True, nifti=False,
                              data_keyword=data_keyword, target_keyword=target_keyword,
                              data_transform=data_transform, target_transform=label_transform)

    dataloader = DataLoader(dataset=trainset, batch_size=2, shuffle=True,
                              num_workers=2, pin_memory=True)
    p = iter(dataloader)
    data, filename= p.next()
