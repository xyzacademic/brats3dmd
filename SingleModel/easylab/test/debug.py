import argparse


parser = argparse.ArgumentParser(description='Train DTS with 2d segmentation')


parser.add_argument('--norm-axis', default='all', type=str, help='Normalization axis')
parser.add_argument('--data', default=0, type=str, help='Data source')
parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
parser.add_argument('--resume', action='store_true', help='Load pre-trained weights')
parser.add_argument('--batch-size', default=4, type=int, help='Batch size.')
parser.add_argument('--seed', default=4096, type=int, help='Random seed.')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate.')
parser.add_argument('--epoch', default=100, type=int, help='The number of epochs')
parser.add_argument('--gpu', default=-1, type=int, nargs='+', help='Using which gpu.')
# parser.add_argument('--threshold', default=0.9, type=float, help='Threshold')
parser.add_argument('--net', default='ours', type=str, help='which network')
parser.add_argument('--loss', default='diceloss', type=str, help='which loss')
parser.add_argument('--schedule', default='s1', type=str, help='Training schedule, s1, s2, s3')
parser.add_argument('--syncbn', action='store_true', help='Using synchronize batchnorm')
parser.add_argument('--basefilter', default=16, type=int, help='Batch size.')
parser.add_argument('--cpu', action='store_true', help='Using CPU')
parser.add_argument('--distributed', action='store_true', help='Whether using distributed training')
parser.add_argument('--comment', default='', type=str, help='comment')
args = parser.parse_args()

a = args.gpu