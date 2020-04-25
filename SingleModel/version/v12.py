import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.normal_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.normal_(m.bias)


global_activation = nn.ReLU()
AFFINE = True


# global_activation = nn.PReLU(num_parameters=1, init=0.25)

class Conv_blocks(nn.Module):
    def __init__(self, channels=16, n_layers=2, kernel_size=3, activation=global_activation, k=3):
        super(Conv_blocks, self).__init__()

        layers = []
        for i in range(n_layers):
            layers += [
                activation,
                # nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=k, bias=False),
                nn.InstanceNorm3d(num_features=channels, eps=1e-5, momentum=0.1, affine=AFFINE),
            ]

        self.convBlock = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        out = self.convBlock(x)

        return x + out


class Downsample(nn.Module):
    def __init__(self, channels=16, kernel_size=3, activation=global_activation, pool=nn.MaxPool3d(2, 2), k=3):
        super(Downsample, self).__init__()
        self.pool = pool
        layers = []
        layers += [
            activation,
            # nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size, stride=2, padding=(kernel_size - 1) // 2, dilation=1, groups=k,
                      bias=False),
            nn.InstanceNorm3d(num_features=channels, eps=1e-5, momentum=0.1, affine=AFFINE),
        ]
        self.conv1x1 = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        out_pool = self.pool(x)
        out_conv = self.conv1x1(x)
        out = torch.stack([out_conv, out_pool], dim=2)
        out = out.view(out.size(0), -1, out.size(3), out.size(4), out.size(5))
        return out


class Upsample(nn.Module):
    def __init__(self, in_channels=16, out_channels=8, kernel_size=3, stride=2, activation=global_activation,
                 bias=False):
        super(Upsample, self).__init__()

        layers = []
        layers += [
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride,
                               bias=bias),
            nn.InstanceNorm3d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=AFFINE),
        ]
        self.upsample = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        return self.upsample(x)


class Encoder(nn.Module):
    '''
    Encoder for shape (192 * 160)
    '''

    def __init__(self, num_channels=3, base_filters=16):
        super(Encoder, self).__init__()
        self.k = num_channels
        self.act = global_activation
        self.conv0 = nn.Conv3d(self.k, base_filters * self.k, 3, padding=1, bias=False, groups=self.k)
        self.bn = nn.InstanceNorm3d(num_features=base_filters * self.k, eps=1e-5, momentum=0.1, affine=AFFINE)
        self.conv1 = Conv_blocks(channels=base_filters * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)
        self.pool1 = Downsample(channels=base_filters * self.k, kernel_size=3, activation=global_activation,
                                pool=nn.MaxPool3d(2, 2), k=self.k)
        self.conv2 = Conv_blocks(channels=base_filters * 2 * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)
        self.pool2 = Downsample(channels=base_filters * 2 * self.k, kernel_size=3, activation=global_activation,
                                pool=nn.MaxPool3d(2, 2), k=self.k)
        self.conv3 = Conv_blocks(channels=base_filters * 4 * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)
        self.pool3 = Downsample(channels=base_filters * 4 * self.k, kernel_size=3, activation=global_activation,
                                pool=nn.MaxPool3d(2, 2), k=self.k)
        self.conv4 = Conv_blocks(channels=base_filters * 8 * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)
        self.pool4 = Downsample(channels=base_filters * 8 * self.k, kernel_size=3, activation=global_activation,
                                pool=nn.MaxPool3d(2, 2), k=self.k)
        self.conv5 = Conv_blocks(channels=base_filters * 16 * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn(out)
        conv1 = self.conv1(out)
        out = self.pool1(conv1)
        conv2 = self.conv2(out)
        out = self.pool2(conv2)
        conv3 = self.conv3(out)
        out = self.pool3(conv3)
        conv4 = self.conv4(out)
        out = self.pool4(conv4)
        conv5 = self.conv5(out)

        return (conv1, conv2, conv3, conv4, conv5)


class Conv_fuse(nn.Module):
    '''
    Combine different mode's feature throught 3D convolution
    '''

    def __init__(self, channels=16, k=3, activation=global_activation):
        super(Conv_fuse, self).__init__()
        layers = []
        se_block = []
        se_block += [
            nn.AdaptiveMaxPool3d(output_size=1),
            nn.Conv3d(in_channels=channels, out_channels=channels // 4, kernel_size=1, bias=False),
            nn.InstanceNorm3d(num_features=channels // 4, eps=1e-5, momentum=0.1, affine=AFFINE),
            # nn.BatchNorm3d(num_features=channels//4, eps=1e-5, momentum=0.1),
            global_activation,
            nn.Conv3d(in_channels=channels // 4, out_channels=channels, kernel_size=1, bias=False),
            # nn.InstanceNorm3d(num_features=channels, eps=1e-5, momentum=0.1),
            nn.Sigmoid()
        ]
        layers += [
            activation,
            # nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels // k, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(num_features=channels // k, eps=1e-5, momentum=0.1, affine=AFFINE),
        ]
        self.se = nn.Sequential(*se_block)
        self.fuse = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        # x_se = F.softmax(self.se(x), dim=1) * x
        x_se = self.se(x) * x
        out = self.fuse(x_se)
        return out


class Decoder(nn.Module):
    '''
    Decoder for shape: (12, 10), (24, 20), (48, 40), (96, 80)
    '''

    def __init__(self, num_classes=2, base_filters=16):
        super(Decoder, self).__init__()
        self.act = global_activation
        self.deconv1 = Upsample(in_channels=base_filters * 16, out_channels=base_filters * 8, kernel_size=2, stride=2,
                                bias=False)
        self.conv1 = Conv_blocks(channels=base_filters * 8, n_layers=2, kernel_size=3, activation=self.act, k=1)

        self.deconv2 = Upsample(in_channels=base_filters * 8, out_channels=base_filters * 4, kernel_size=2, stride=2,
                                bias=False)
        self.conv2 = Conv_blocks(channels=base_filters * 4, n_layers=2, kernel_size=3, activation=self.act, k=1)

        self.deconv3 = Upsample(in_channels=base_filters * 4, out_channels=base_filters * 2, kernel_size=2, stride=2,
                                bias=False)
        self.conv3 = Conv_blocks(channels=base_filters * 2, n_layers=2, kernel_size=3, activation=self.act, k=1)

        self.deconv4 = Upsample(in_channels=base_filters * 2, out_channels=base_filters, kernel_size=2, stride=2,
                                bias=False)
        self.conv4 = Conv_blocks(channels=base_filters, n_layers=2, kernel_size=3, activation=self.act, k=1)

        # self.conv = nn.Conv3d(in_channels=base_filters, out_channels=num_classes, kernel_size=1, stride=1, padding=0,
        #                       bias=True)
        self.apply(_weights_init)

    def forward(self, features):
        assert len(features) == 5
        f4, f3, f2, f1, f0 = features  # inverse variables' name order

        out1 = self.deconv1(f0)
        out1 = f1 + out1  # multiply encoder's feature with related output of deconvolution layer.
        out1 = self.conv1(out1)

        out2 = self.deconv2(out1)
        out2 = f2 + out2
        out2 = self.conv2(out2)

        out3 = self.deconv3(out2)
        out3 = f3 + out3
        out3 = self.conv3(out3)

        out4 = self.deconv4(out3)
        out4 = f4 + out4
        out4 = self.conv4(out4)

        # out4 = F.relu(out4)

        return out4


class ModalilyFuse(nn.Module):
    def __init__(self, base_filters=16, k=3):
        super(ModalilyFuse, self).__init__()
        self.act = global_activation
        self.k = k
        self.fuse0 = Conv_fuse(channels=base_filters * self.k, k=self.k, activation=self.act)
        self.fuse1 = Conv_fuse(channels=base_filters * 2 * self.k, k=self.k, activation=self.act)
        self.fuse2 = Conv_fuse(channels=base_filters * 4 * self.k, k=self.k, activation=self.act)
        self.fuse3 = Conv_fuse(channels=base_filters * 8 * self.k, k=self.k, activation=self.act)
        self.fuse4 = Conv_fuse(channels=base_filters * 16 * self.k, k=self.k, activation=self.act)
        self.apply(_weights_init)

    def forward(self, inputs):
        assert len(inputs) == 5
        x0, x1, x2, x3, x4 = inputs

        f0 = self.fuse0(x0)
        f1 = self.fuse1(x1)
        f2 = self.fuse2(x2)
        f3 = self.fuse3(x3)
        f4 = self.fuse4(x4)
        features = (f0, f1, f2, f3, f4)

        return features


class MPN(nn.Module):
    '''
    Net work used to do segmentation.
    Input shape: (batch_size, mode, 192, 160)
    Output shape:
    '''

    def __init__(self, num_channels=3, base_filters=16, num_classes=2, norm_axis='all'):
        super(MPN, self).__init__()
        self.num_classes = num_classes
        self.norm_axis = norm_axis
        self.act = global_activation
        self.k = num_channels
        self.dropout = nn.Dropout3d(0.25)
        self.conv0 = nn.Conv3d(self.k*2, base_filters * self.k, 3, padding=1, bias=False, groups=self.k*2)
        self.bn = nn.InstanceNorm3d(num_features=base_filters * self.k, eps=1e-5, momentum=0.1, affine=AFFINE)
        self.shortconnect = Conv_blocks(channels=base_filters * self.k, n_layers=2, kernel_size=3,
                                        activation=nn.ReLU(inplace=True), k=self.k*2)
        self.encoder = Encoder(num_channels=num_channels, base_filters=base_filters)
        self.et_fuse = ModalilyFuse(base_filters=base_filters, k=num_channels)
        self.et_decoder = Decoder(num_classes=num_classes, base_filters=base_filters)
        self.ncr_fuse = ModalilyFuse(base_filters=base_filters, k=num_channels)
        self.ncr_decoder = Decoder(num_classes=num_classes, base_filters=base_filters)
        self.ed_fuse = ModalilyFuse(base_filters=base_filters, k=num_channels)
        self.ed_decoder = Decoder(num_classes=num_classes, base_filters=base_filters)
        self.et_conv = nn.Conv3d(in_channels=base_filters, out_channels=2, kernel_size=1, stride=1, padding=0,
                                 bias=True)
        self.ed_conv = nn.Conv3d(in_channels=base_filters, out_channels=2, kernel_size=1, stride=1, padding=0,
                                 bias=True)
        self.ncr_conv = nn.Conv3d(in_channels=base_filters, out_channels=2, kernel_size=1, stride=1, padding=0,
                                 bias=True)
        self.wt_conv = nn.Conv3d(in_channels=base_filters * 3, out_channels=2, kernel_size=1, stride=1, padding=0,
                                 bias=True)
        self.tc_conv = nn.Conv3d(in_channels=base_filters * 2, out_channels=2, kernel_size=1, stride=1, padding=0,
                                 bias=True)
        self.final_conv = nn.Conv3d(in_channels=base_filters * (5+3+self.k), out_channels=num_classes, kernel_size=1, stride=1,
                                    padding=0,
                                    bias=True)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.dropout(x)
        normal = x[:, :4]

        features = self.encoder(normal)
        et = self.et_fuse(features)  # 4
        ed = self.ed_fuse(features)  # 2
        ncr = self.ncr_fuse(features)  # 1

        et = self.et_decoder(et)
        ed = self.ed_decoder(ed)
        ncr = self.ncr_decoder(ncr)
        et_out = self.et_conv(et)
        et_prob = F.softmax(et_out, dim=1)[:, 1:2]
        ed_out = self.ed_conv(ed)
        ed_prob = F.softmax(ed_out, dim=1)[:, 1:2]
        ncr_out = self.ncr_conv(ncr)
        ncr_prob = F.softmax(ncr_out, dim=1)[:, 1:2]
        wt = torch.cat([et, ed, ncr], dim=1)
        wt_out = self.wt_conv(wt)
        wt_prob = F.softmax(wt_out, dim=1)[:, 1:2]
        tc = torch.cat([et, ncr], dim=1)
        tc_out = self.tc_conv(tc)
        tc_prob = F.softmax(tc_out, dim=1)[:, 1:2]
        short_out = self.conv0(x)
        short_out = self.bn(short_out)
        short_out = self.shortconnect(short_out)
        # short_out = F.relu(short_out, inplace=True)

        # out = torch.cat([et*(1+et_prob), ed*(1+ed_prob), ncr*(1+ncr_prob), wt*(1+wt_prob), tc*(1+tc_prob),
        #                  short_out*(1+wt_prob)], dim=1)
        out = torch.cat([et, ed, ncr, wt, tc,
                         short_out], dim=1) * wt_prob
        out = self.final_conv(out)

        return out, wt_out, tc_out, et_out, ed_out, ncr_out