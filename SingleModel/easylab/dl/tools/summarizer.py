#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""summarizer

Define a class "Summarizer" to account network structure's parameters
and FLOPs

Reference:
Link:

Updating log:
V1.0 description
"""

import torch
import torch.nn as nn
from collections import OrderedDict


default_summary_modules = [
    'Conv3d',
    'Conv2d'
    'Add',
    'Linear',
    'ReLU',
    'BatchNorm3d',
    'InstanceNorm3d'
    'MaxPool2d',
    'AvgPool2d',
    'Upsample',
    'ConvTranspose3d',
]  # module list


class Summarizer(object):
    def __init__(self, summary_modules=default_summary_modules):
        self._hooks = []
        self._info = OrderedDict()
        self._summary_modules = summary_modules
        self._id_to_name = {}

    def _register_hook(self, module):
        handle = module.register_forward_hook(self._summary_hook)
        self._hooks.append(handle)

    def _summary_hook(self, module, input, output):
        r"""
        hook function
        :param module: pytorch module
        :param input: input of this module
        :param output: output of this module
        :return:
        """
        module_type = module.__class__.__name__
        if module_type not in self._summary_modules:
            return

        name = self._id_to_name[str(id(module))]
        self._info[name] = OrderedDict()

        self._info[name]['type'] = module_type
        self._info[name]['input_shape'] = list(input[0].size())
        self._info[name]['output_shape'] = list(output[0].size())

        param_num = 0
        flops = 0
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, nn.Conv3d):
                _, _, output_height, output_width, output_depth = output.size()
                groups = 1
                if hasattr(module, 'groups'):
                    groups = module.groups
                output_channel, input_channel, kernel_height, kernel_width, kernel_depth = module.weight.size()
                flops = 1.0/groups * output_channel * output_height * output_width * input_channel * kernel_height * \
                        kernel_width * kernel_depth

            if isinstance(module, nn.Linear):
                input_num, output_num = module.weight.size()
                flops = input_num * output_num

            param_num += module.weight.numel()
            self._info[name]['trainable'] = module.weight.requires_grad
            self._info[name]['weight'] = list(module.weight.size())
        else:
            self._info[name]['weight'] = 'None'

        if hasattr(module, 'bias') and module.bias is not None:
            param_num += module.bias.numel()
            flops += module.bias.numel()

        self._info[name]['param_num'] = param_num
        self._info[name]['flops'] = flops

    def summarize(self, model, *args, **kwargs):
        for name, module in model.named_modules():
            self._id_to_name[str(id(module))] = name

        # register hook
        model.apply(self._register_hook)
        model(*args, **kwargs)

        print('========================================================================================================================')
        line_new = '{:>30} {:>15} {:>25} {:>25} {:>25} {:>25} {:>25}'.format('Layer', 'Type', 'Input Shape',
                                                                      'Output Shape', 'Weight_Shape',
                                                              'Params #', 'FLOPS #')
        print(line_new)
        print('========================================================================================================================')
        total_params = 0
        trainable_params = 0
        total_flops = 0
        for name, info in self._info.items():
            # input_shape, output_shape, trainable, nb_params
            line_new = '{:>30} {:>15} {:>25}  {:>25} {:>25} {:>25} {:>20}'.format(name, info['type'],
                                                                                 str(info['input_shape']), str(info['output_shape']), str(info['weight']), '{0:,}'.format(info['param_num']), '{0:,}'.format(info['flops']))
            total_params += info['param_num']
            if 'trainable' in info and info['trainable'] == True:
                trainable_params += info['param_num']
            total_flops += info['flops']
            print(line_new)
        print('=======================================================================================================================')
        print('Total params: {0:,}'.format(total_params))
        print('Trainable params: {0:,}'.format(trainable_params))
        print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
        print('Total FLOPS: {0:,}'.format(total_flops))
        print('=======================================================================================================================')

        # clear instance information
        self.clear()

    def clear(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._info.clear()
        self._id_to_name.clear()


if __name__ == '__main__':
    pass
