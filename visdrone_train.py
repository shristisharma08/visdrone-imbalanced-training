print(" Running visdrone_train.py from:", __file__)
import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import prepare_folders, adjust_learning_rate, train, validate, save_checkpoint
from losses import IBLoss, FocalLoss, LDAMLoss
from imbalance_visdrone import IMBALANCEVisDrone
from visdrone_dataset import VisDroneDataset
from opts import parser
print(" LOADED PARSER FROM opts.py WITH args:", parser._option_string_actions.keys())

args = parser.parse_args()
print("ARGS:", vars(args))  

best_acc1 = 0

def main():

    args.store_name = '_'.join([
        args.dataset, args.arch, args.loss_type, args.train_rule, 
        args.imb_type, str(args.imb_factor), args.exp_str
    ])
    args.device = torch.device("cpu")
    args.weight_decay = 5e-4
    args.momentum = 0.9
    args.lr = 0.1
    args.batch_size = 16  
    args.workers = 0
    args.epochs = 200
    args.start_epoch = 0
    args.root_log = './logs'
    args.root_model = './models'
    prepare_folders(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    main_worker(args)

def main_worker(args):
    global best_acc1

    print("=> Creating model 'ResNet44'—VisDrone’s aerial ace!")
    num_classes = 12  
    model = models.ResNet44(num_classes=num_classes)

    model = model.to(args.device)
    print(f"Using device: {args.device}—keeping it sky-high!")

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    root = r"C:\Users\csio\Desktop\Deep Learning\IB-Loss-main"

    print(f" Loading VisDrone from: {root}")
    full_dataset = VisDroneDataset(root=root, train=True, transform=transform_train)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = transform_val

    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty! Check path: {root}")

    print(f" Loaded VisDrone with {len(train_dataset)} training images—time to fly!")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    print(f" Train loader ready with {len(train_loader)} batches")

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers)

    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        print(f" Starting training for epoch {epoch}...")
        train(train_loader, model, optimizer, epoch, args, log_training, tf_writer)
        print(f" Training done for epoch {epoch}, now validating...")
        acc1 = validate(val_loader, model, epoch, args, log_testing, tf_writer)
        print(f" Validation accuracy for epoch {epoch}: {acc1:.2f}%")
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': 'ResNet44',
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

if __name__ == '__main__':
    main()
