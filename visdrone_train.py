import os
import sys
import opts
parser = opts.get_parser()
args = parser.parse_args()
print("Parsed Arguments:", args)
import torch
import models
from torchvision import transforms
from models.resnet_visdrone import ResNet44
from utils import *
from losses import IBLoss, FocalLoss, LDAMLoss
from imbalance_visdrone import IMBALANCEVisDrone
from visdrone_dataset import VisDroneDataset
from torch.utils.tensorboard import SummaryWriter


# Logger class for saving all prints
class Logger(object):
    def __init__(self, console, file):
        self.console = console
        self.file = file

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

def main():
    print("Inside main() function")
    if not args:
        raise ValueError("No arguments passed!")

    args.store_name = '_'.join([
        args.dataset, args.arch, args.loss_type, args.train_rule,
        args.imb_type, str(args.imb_factor), args.exp_str
    ])
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

   
    os.makedirs(args.root_log, exist_ok=True)
    log_file = open(os.path.join(args.root_log, f"{args.exp_str}_results.txt"), "w")
    sys.stdout = Logger(sys.stdout, log_file)

    if args.arch == 'resnet44':
        model = ResNet44(num_classes=args.num_classes).to(args.device)
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes).to(args.device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )

    if args.loss_type == 'IBLoss':
        criterion = IBLoss()
    elif args.loss_type == 'FocalLoss':
        criterion = FocalLoss()
    elif args.loss_type == 'LDAMLoss':
        criterion = LDAMLoss()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    root = r"C:\Users\csio\Desktop\Deep Learning\IB-Loss-main"
    print(f"Loading VisDrone dataset from: {root}")
    full_dataset = VisDroneDataset(root=root, train=True, transform=transform_train)

    if args.imb_type == 'exp':
        imbalance = IMBALANCEVisDrone(
            root="C:/Users/csio/Desktop/Deep Learning/IB-Loss-main/VisDrone2019-DET-train/organized",
            imb_type='exp',
            imb_factor=args.imb_factor,
            rand_number=args.rand_number,
            transform=transform_train
        )
    elif args.imb_type == 'step':
        imbalance = IMBALANCEVisDrone(
            imb_type='step',
            imb_factor=args.imb_factor,
            full_dataset=full_dataset
        )
    else:
        raise ValueError(f"Unsupported imbalance type: {args.imb_type}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty!")

    print(f"Train set size: {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False, num_workers=0
    )

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    best_acc1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch} ---")
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, optimizer, epoch, args, tf_writer=tf_writer)
        acc1 = validate(val_loader, model, epoch, args, tf_writer=tf_writer)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    
    log_file.close()


if __name__ == '__main__':
    print("Script execution started.")
    main()
