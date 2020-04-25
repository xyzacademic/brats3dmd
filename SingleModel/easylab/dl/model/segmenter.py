# build-in library
import os
from time import time

# third-party library
import numpy as np
import torch
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F
from torch import nn
import torch.utils.data
import pandas as pd
import apex
from apex import amp


from easylab.dl.log import record_csv
from easylab.dl.utils import AverageMeter
from easylab.dl.utils import evaluation
from easylab.dl.model.model import Model
from easylab.dl.model.model import reduce_tensor


class Brats19Segmenter(Model):

    def train(self, epoch=0, train_loader=None, em_list=None, verbose=True):
        config = self.config
        log_columns = ['epoch', 'loss'] + em_list
        record_csv(epoch, log_columns, config['save_path'], "train.csv")
        print("\nEpoch: %d" % epoch)
        self.net.train()
        train_loss = AverageMeter('train_loss')
        eval_dict = {}
        if em_list is not None:
            for method in em_list:
                eval_dict[method] = AverageMeter(method)

        start = time()

        for batch_idx, (data, target) in enumerate(train_loader):

            if config['use_cuda']:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()
            outputs = self.net(data)
            loss = self.criterion(outputs, target)
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            if config['distributed']:
                train_loss.update(reduce_tensor(loss.data).item(),
                                  target.size(0))
            else:
                train_loss.update(loss.item(), target.size(0))
            if em_list is not None:
                # predict = outputs.max(1)[1]
                predict = F.softmax(outputs, dim=1).round()
                # predict = torch.sigmoid(outputs).round()
                value = evaluation(predict, target, 'dice_brats19')
                for i in range(len(em_list)):
                    if config['distributed']:
                        eval_dict[em_list[i]].update(reduce_tensor(
                            value.mean(dim=0)[i]).item(),
                                                 target.size(0))
                    else:
                        eval_dict[em_list[i]].update(value.mean(dim=0)[i].item(),
                                                 target.size(0))

        df = {}
        df['epoch'] = epoch
        print('%s: %.5f' % (train_loss.name, train_loss.avg))
        df['loss'] = train_loss.avg
        print("This epoch cost %.2f seconds" % (time() - start))
        if em_list is not None:
            for i in range(len(em_list)):
                print("%s: %.5f" % (em_list[i], eval_dict[em_list[i]].avg))
                df[em_list[i]] = eval_dict[em_list[i]].avg
        record_csv(epoch, df, config['save_path'], "train.csv")

        return train_loss.avg

    def test(self, epoch=0, test_loader=None, em_list=None, verbose=True):
        config = self.config
        log_columns = ['epoch', 'loss'] + em_list
        record_csv(epoch, log_columns, config['save_path'], "test.csv")
        self.net.eval()
        test_loss = AverageMeter('test_loss')
        eval_dict = {}
        if em_list is not None:
            for method in em_list:
                eval_dict[method] = AverageMeter(method)

        start = time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if config['use_cuda']:
                    data, target = data.cuda(), target.cuda()

                outputs = self.net(data)
                loss = self.criterion(outputs, target)
                if config['distributed']:
                    test_loss.update(reduce_tensor(loss.data).item(),
                                     target.size(0))
                else:
                    test_loss.update(loss.item(), target.size(0))
                if em_list is not None:
                    # predict = outputs.max(1)[1]
                    predict = F.softmax(outputs, dim=1).round()
                    # predict = torch.sigmoid(outputs).round()
                    value = evaluation(predict, target, 'dice_brats19')
                    for i in range(len(em_list)):
                        if config['distributed']:
                            eval_dict[em_list[i]].update(reduce_tensor(
                                value.mean(dim=0)[i]).item(),
                                                        target.size(0))
                        else:
                            eval_dict[em_list[i]].update(value.mean(dim=0)[i].item(),
                                                        target.size(0))
            df = {}
            df['epoch'] = epoch
            print('%s: %.5f' % (test_loss.name, test_loss.avg))
            df['loss'] = test_loss.avg
            print("This epoch cost %.2f seconds" % (time() - start))
            if em_list is not None:
                for i in range(len(em_list)):
                    print("%s: %.5f" % (em_list[i], eval_dict[em_list[i]].avg))
                    df[em_list[i]] = eval_dict[em_list[i]].avg
            record_csv(epoch, df, config['save_path'], "test.csv")

        return test_loss.avg

    def inference(self, test_loader=None, prob=False, target_path=None, replace_origin='anat', replace_target='pred'):
        config = self.config
        assert self.net is not None
        root = self.test_loader.dataset.root
        self.net.eval()
        start = time()

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        with torch.no_grad():
            for batch_idx, (data, filenames) in enumerate(test_loader):
                if config['use_cuda'] is True:
                    data = data.cuda()

                outputs = self.net(data)
                if prob is True:
                    pre = F.softmax(outputs, dim=1)[:, 1]
                else:
                    pre = outputs.max(1)[1].data.cpu().numpy()

                for i in range(pre.shape[0]):
                    patient = nib.load(os.path.join(root, filenames[i]))
                    new_pre = np.zeros(shape=(240, 240, 155), dtype=np.float32)
                    new_pre[:, 38:198, 28:220, :] = pre[:, :, :, 5:]
                    new_data = nib.Nifti1Image(pre[i], patient.affine, patient.header)
                    nib.save(new_data, os.path.join(target_path, filenames[i].replace(replace_origin, replace_target)))
                    print('Save {file} successfully.'.format(file=os.path.join(target_path,
                                                                               filenames[i].replace('anat', 'pred'))))
        print('This inference cost %.2f seconds' % (time() - start))
