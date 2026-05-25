import os
import random
import logging
from PIL import ImageFile, Image
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from utils.logger import setup_logger
from utils.utils import save_checkpoint
from utils.optimizers import configure_optimizers
from utils.training import train_one_epoch
from utils.testing import test_one_epoch
from loss.rd_loss import RateDistortionLoss
from config.args import train_options
from config.config import model_config
from models import ELIC
import random


def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()

    # --- DISPATCHER TYPE-CASTING FIX ---
    # Convert CLI strings back into the math types PyTorch expects
    args.learning_rate = float(args.learning_rate)
    args.aux_learning_rate = float(args.aux_learning_rate)
    args.lmbda = float(args.lmbda)
    args.clip_max_norm = float(args.clip_max_norm)
    args.batch_size = int(args.batch_size)
    args.test_batch_size = int(args.test_batch_size)
    args.epochs = int(args.epochs)
    # -----------------------------------

    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        random.seed(seed)

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))

    setup_logger('train', os.path.join('./experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join('./experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('./experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('./experiments', args.experiment, 'checkpoints'))

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    # get names from args

    # Safely grab the explicit overrides passed to script
    explicit_train = getattr(args, 'train_dataset', None)
    explicit_test = getattr(args, 'test_dataset', None)
    
    train_split = getattr(args, 'train_split', '')
    test_split = getattr(args, 'test_split', '')

    # Determine Training Path: Use explicit path if provided, otherwise fallback to using dataset root + split variables
    if explicit_train:
        train_path = explicit_train
    else:
        train_path = os.path.join(args.dataset, train_split) if train_split else args.dataset

    # Determine Test Path: Use explicit path if provided, otherwise fallback to using dataset root + split variables
    if explicit_test:
        test_path = explicit_test
    else:
        test_path = os.path.join(args.dataset, test_split) if test_split else args.dataset    
    
    # Pass split="" so ImageFolder reads directly from the exact paths we just built
    train_dataset = ImageFolder(train_path, split="", transform=train_transforms)
    test_dataset = ImageFolder(test_path, split="", transform=test_transforms)

    # Dataloaders remain exactly the same
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    
    print(f"DEBUG: Found {len(train_dataset)} training images.")
    print(f"DEBUG: Found {len(test_dataset)} validation images.")
    if len(test_dataset) == 0:
        raise RuntimeError(f"Validation dataset 'kodak' is empty! Check path: {args.dataset}")


    net = ELIC(config=config)
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)

    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450,550], gamma=0.1)
        lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
        lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
        # print(lr_scheduler.state_dict())
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_dataloader) / args.batch_size)
        checkpoint = None
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    # start_epoch = 0
    # best_loss = 1e10
    # current_step = 0

    logger_train.info(f"Seed: {seed}")
    logger_train.info(args)
    # logger_train.info(net)
    logger_train.info(optimizer)
    logger_train.info(aux_optimizer)
    optimizer.param_groups[0]['lr'] = args.learning_rate
    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step
        )

        save_dir = os.path.join('./experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
        loss = test_one_epoch(epoch, test_dataloader, net, criterion, save_dir, logger_val, tb_logger)
        # lr_scheduler.step(loss)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        net.update(force=True)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                os.path.join('./experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')

if __name__ == '__main__':
    main()
