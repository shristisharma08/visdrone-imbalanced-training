print("✅ Loaded opts.py from:", __file__)

import argparse

parser = argparse.ArgumentParser(description='PyTorch VisDrone Training')


parser.add_argument('--dataset', default='visdrone', help='dataset setting (default: VisDrone)')
parser.add_argument('-a', '--arch', default='resnet44', help='model architecture (default: ResNet-44)')
parser.add_argument('--num_classes', default=12, type=int, help='number of classes (VisDrone has 12!)')
parser.add_argument('--loss_type', default="IBLoss", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy')
parser.add_argument('--start_ib_epoch', default=100, type=int, help='start epoch for IB Loss')
parser.add_argument('--rand_number', default=0, type=int, help='random seed')
parser.add_argument('--exp_str', default='0', type=str, help='experiment ID')
parser.add_argument('-j', '--workers', default=4, type=int, help='data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='total training epochs')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=16, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=2e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--evaluate', action='store_true', help='evaluate model only')
parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for reproducibility')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('--root_log', type=str, default='log', help='log directory')
parser.add_argument('--root_model', type=str, default='checkpoint', help='model save path')


