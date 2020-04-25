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
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        # print(m)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        # print(m)
        m.weight.data.fill_(1)
        m.bias.data.zero_()


global_activation = nn.ReLU(inplace=True)
AFFINE = True

class ConvBlocks(nn.Module):
    def __init__(self, channels=16, n_layers=2, kernel_size=3, activation=global_activation, k=3):
        super(ConvBlocks, self).__init__()

        layers = []
        for i in range(n_layers):
            layers += [
                nn.InstanceNorm3d(num_features=channels, eps=1e-5, momentum=0.1, affine=AFFINE),
                # nn.GroupNorm(num_groups=channels//16, num_channels=channels),
                # nn.BatchNorm3d(num_features=channels, eps=1e-5, momentum=0.1),
                activation,
                nn.Conv3d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=k, bias=False),
            ]

        self.convBlock = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        out = self.convBlock(x)

        return x + out

class DownSampling(nn.Module):
    def __init__(self, channels=16, kernel_size=3, activation=global_activation, pool=nn.MaxPool3d(2, 2), k=3):
        super(DownSampling, self).__init__()
        self.pool = pool
        layers = []
        layers += [
            nn.InstanceNorm3d(num_features=channels, eps=1e-5, momentum=0.1, affine=AFFINE),
            # nn.GroupNorm(num_groups=channels // 16, num_channels=channels),
            # nn.BatchNorm3d(num_features=channels, eps=1e-5, momentum=0.1),
            activation,
            nn.Conv3d(channels, channels, kernel_size, stride=2, padding=(kernel_size - 1) // 2, dilation=1, groups=k,
                      bias=False)
        ]
        self.conv1x1 = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        out_pool = self.pool(x)
        out_conv = self.conv1x1(x)
        out = torch.stack([out_conv, out_pool], dim=2)
        out = out.view(out.size(0), -1, out.size(3), out.size(4), out.size(5))
        return out


class UpSampling(nn.Module):
    def __init__(self, in_channels=16, out_channels=8, kernel_size=3, stride=2, activation=global_activation,
                 bias=False):
        super(UpSampling, self).__init__()

        layers = []
        layers += [
            nn.InstanceNorm3d(num_features=in_channels, eps=1e-5, momentum=0.1, affine=AFFINE),
            # nn.GroupNorm(num_groups=channels // 16, num_channels=channels),
            # nn.BatchNorm3d(num_features=in_channels, eps=1e-5, momentum=0.1),
            # activation,
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride,
                               bias=bias)
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
        self.conv1 = ConvBlocks(channels=base_filters * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)
        self.pool1 = DownSampling(channels=base_filters * self.k, kernel_size=3, activation=global_activation,
                                pool=nn.AvgPool3d(2, 2), k=self.k)
        self.conv2 = ConvBlocks(channels=base_filters * 2 * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)
        self.pool2 = DownSampling(channels=base_filters * 2 * self.k, kernel_size=3, activation=global_activation,
                                pool=nn.AvgPool3d(2, 2), k=self.k)
        self.conv3 = ConvBlocks(channels=base_filters * 4 * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)
        self.pool3 = DownSampling(channels=base_filters * 4 * self.k, kernel_size=3, activation=global_activation,
                                pool=nn.AvgPool3d(2, 2), k=self.k)
        self.conv4 = ConvBlocks(channels=base_filters * 8 * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)
        self.pool4 = DownSampling(channels=base_filters * 8 * self.k, kernel_size=3, activation=global_activation,
                                pool=nn.AvgPool3d(2, 2), k=self.k)
        self.conv5 = ConvBlocks(channels=base_filters * 16 * self.k, n_layers=2, kernel_size=3, activation=self.act,
                                 k=self.k)

        self.apply(_weights_init)

    def forward(self, x):
        # x = self.input_bn(x)
        out = self.conv0(x)
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


class ConvFusion(nn.Module):
    '''
    Combine different mode's feature throught 3D convolution
    '''

    def __init__(self, channels=16, k=3, activation=global_activation):
        super(ConvFusion, self).__init__()
        layers = []
        se_block = []
        se_block += [
            nn.AdaptiveMaxPool3d(output_size=1),
            nn.Conv3d(in_channels=channels, out_channels=channels//4, kernel_size=1, bias=False),
            nn.InstanceNorm3d(num_features=channels//4, eps=1e-5, momentum=0.1, affine=AFFINE),
            global_activation,
            nn.Conv3d(in_channels=channels//4, out_channels=channels, kernel_size=1, bias=True),
            # nn.InstanceNorm3d(num_features=channels, eps=1e-5, momentum=0.1),
            nn.Sigmoid()
        ]
        layers += [
            nn.InstanceNorm3d(num_features=channels, eps=1e-5, momentum=0.1, affine=AFFINE),
            # nn.GroupNorm(num_groups=channels // 16, num_channels=channels),
            # nn.BatchNorm3d(num_features=channels, eps=1e-5, momentum=0.1),
            activation,
            nn.Conv3d(channels, channels // k, kernel_size=1, stride=1, bias=False)
        ]
        self.se = nn.Sequential(*se_block)
        self.fuse = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
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
        self.deconv1 = UpSampling(in_channels=base_filters * 16, out_channels=base_filters * 8, kernel_size=2, stride=2,
                                bias=False)
        self.conv1 = ConvBlocks(channels=base_filters * 8, n_layers=2, kernel_size=3, activation=self.act, k=1)

        self.deconv2 = UpSampling(in_channels=base_filters * 8, out_channels=base_filters * 4, kernel_size=2, stride=2,
                                bias=False)
        self.conv2 = ConvBlocks(channels=base_filters * 4, n_layers=2, kernel_size=3, activation=self.act, k=1)

        self.deconv3 = UpSampling(in_channels=base_filters * 4, out_channels=base_filters * 2, kernel_size=2, stride=2,
                                bias=False)
        self.conv3 = ConvBlocks(channels=base_filters * 2, n_layers=2, kernel_size=3, activation=self.act, k=1)

        self.deconv4 = UpSampling(in_channels=base_filters * 2, out_channels=base_filters, kernel_size=2, stride=2,
                                bias=False)
        self.conv4 = ConvBlocks(channels=base_filters, n_layers=2, kernel_size=3, activation=self.act, k=1)

        self.conv = nn.Conv3d(in_channels=base_filters, out_channels=num_classes, kernel_size=1, stride=1, padding=0,
                              bias=True)
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

        # out4 = self.bn(out4)
        # out4 = self.relu(out4)

        out = self.conv(out4)

        return out


class ModalityFuse(nn.Module):
    def __init__(self, base_filters=16, k=3):
        super(ModalityFuse, self).__init__()
        self.act = global_activation
        self.k = k
        self.fuse0 = ConvFusion(channels=base_filters * self.k, k=self.k, activation=self.act)
        self.fuse1 = ConvFusion(channels=base_filters * 2 * self.k, k=self.k, activation=self.act)
        self.fuse2 = ConvFusion(channels=base_filters * 4 * self.k, k=self.k, activation=self.act)
        self.fuse3 = ConvFusion(channels=base_filters * 8 * self.k, k=self.k, activation=self.act)
        self.fuse4 = ConvFusion(channels=base_filters * 16 * self.k, k=self.k, activation=self.act)
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
    Multi-Path network architecture with full pre-activation and SE-block used to do segmentation.
    Input shape: (batch_size, mode, 192, 160)
    Output shape:
    '''

    def __init__(self, num_channels=3, base_filters=16, num_classes=2, norm_axis='all'):
        super(MPN, self).__init__()
        self.num_classes = num_classes
        self.norm_axis = norm_axis
        self.act = global_activation
        self.encoder = Encoder(num_channels=num_channels, base_filters=base_filters)
        self.fuse = ModalityFuse(base_filters=base_filters, k=num_channels)
        self.decoder = Decoder(num_classes=num_classes, base_filters=base_filters)

        self.apply(_weights_init)

    def forward(self, x):
        # x = torch.cat([x, x.flip(2)], dim=1)
        features = self.encoder(x)
        features = self.fuse(features)
        out = self.decoder(features)

        return out


if __name__ == '__main__':
    x = torch.randn((1, 3, 160, 192, 160)).cuda().half()
    label = torch.randint(0, 2, (1, 160, 192, 160)).long().cuda()
    from lossfunction import DiceLoss

    criterion = DiceLoss(reduce='mean')
    net = DTS()
    net.cuda()
    # y1 = net(x)
    # l1 = criterion(y1, label)
    net.half()
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
            layer.float()
    y = net(x)