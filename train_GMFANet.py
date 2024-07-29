import argparse
import os
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
from models.__init__ import *
from models.GMFANet import GMFANet
import torch.nn.functional as F
from utils import GradientGuidance
os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser(description='Training of GMFANet in pytorch')
parser.add_argument('--n_classes', '-n', metavar='N', type=int, default='21',)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='/home/GMFANet/checkpoint', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

if torch.cuda.is_available():
    print("cuda prepared!")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
best_prec1 = 0
best_prec5=0

import torch
import torch.nn as nn

def main():
    global args, best_prec1, best_prec5
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = GMFANet(n_classes=args.n_classes, device=device, pretrained=True)
    model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec5=checkpoint['best_prec5']
            model.load_state_dict(checkpoint['state_dict'])
            args.lr=checkpoint['lr']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root='/home/datasets/UCMerced_LandUse/80/train_data',
            transform=transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root='/home/datasets/UCMerced_LandUse/80/test_data',
            transform=transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD([{'params':model.parameters(),'initial_lr':args.lr}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[80, 100], last_epoch=args.start_epoch - 1)
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()
        prec1, prec5 = validate(val_loader, model, criterion, epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = max(prec5, best_prec5)
        if is_best:
            best_epoch = epoch


        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'epoch':epoch,
        }, is_best, filename=os.path.join(args.save_dir, 'model_GMFANet.th'))

    print(best_prec1,best_prec5,best_epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    mse_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5=AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        mc=copy.deepcopy(model)
        mc.tau=1
        mc.target_att = torch.ones(input.shape[0], 1, 7, 7).to(device)
        gradientguidance = GradientGuidance(model=mc, target_layers=[mc.MultiScaleAttention4], device=device, use_cuda=True)
        grayscale, weights = gradientguidance(input_tensor=input_var, target_category=target_var)
        grayscale = torch.Tensor(grayscale).unsqueeze(1).to(device)
        target_att = nn.AdaptiveAvgPool2d((7, 7))(grayscale)
        model.target_att = target_att.clone()
        alpha = 1
        model.tau = cal_t(0.3, 0.1, epoch//10)

        output, att_pred= model(input_var)
        cla_loss = criterion(output, target_var)
        mse_loss = F.mse_loss(att_pred, target_att)
        # compute gradient and do SGD step
        loss = cla_loss+alpha*mse_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        cla_loss = cla_loss.float()
        mse_loss = mse_loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target.data)[0]
        prec5 = accuracy(output.data, target.data)[1]
        losses.update(loss.item(), input.size(0)) #input.size(0)
        cla_losses.update(cla_loss.item(), input.size(0))
        mse_losses.update(mse_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0)) #input.size(0)
        top5.update(prec5.item(), input.size(0)) #input.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Cla Loss {cla_loss.val:.4f} ({cla_loss.avg:.4f})\t'
                  'Mse Loss {mse_loss.val:.4f} ({mse_loss.avg:.4f})\t'
                  'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
            'Tau {tau_model:.1f} ({tau_mc:.1f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, cla_loss=cla_losses, mse_loss=mse_losses, loss=losses, top1=top1, top5=top5, tau_model=model.tau, tau_mc=mc.tau))
    return loss

def validate(val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    mse_losses = AverageMeter()
    cla_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to evaluate mode
    model.eval()
    mc = copy.deepcopy(model)
    mc.tau=1
    mc.target_att = torch.ones(args.batch_size, 1, 7, 7).to(device)
    gradientguidance = GradientGuidance(model=mc, target_layers=[mc.MultiScaleAttention4], device=device, use_cuda=True)
    end = time.time()
    # with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input_var = input.to(device)
        target_var = target.to(device)

        # compute output
        grayscale, weights = gradientguidance(input_tensor=input_var, target_category=target_var)
        grayscale = torch.Tensor(grayscale).unsqueeze(1).to(device)
        target_att = nn.AdaptiveAvgPool2d((7, 7))(grayscale)
        model.target_att = target_att.clone()
        model.tau = cal_t(0.3, 0.1, epoch // 10)

        output, att_pred = model(input_var)
        cla_loss = criterion(output, target_var)
        mse_loss = F.mse_loss(att_pred, target_att)
        loss = cla_loss+mse_loss


        output = output.float()
        cla_loss = cla_loss.float()
        loss = loss.float()
        mse_loss = mse_loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        prec5 = accuracy(output.data, target)[1]
        losses.update(loss.item(), input.size(0))
        cla_losses.update(cla_loss.item(), input.size(0))
        mse_losses.update(mse_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Cla Loss {cla_loss.val:.4f} ({cla_loss.avg:.4f})\t'
                  'Mse Loss {mse_loss.val:.4f} ({mse_loss.avg:.4f})\t'
                  'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
            'Tau {tau_model:.1f} ({tau_mc:.1f})'.format(
                      i, len(val_loader), batch_time=batch_time, cla_loss=cla_losses, mse_loss=mse_losses, loss=losses,
                      top1=top1, top5=top5, tau_model=model.tau, tau_mc=mc.tau))

    print(' * Prec@1 {top1.avg:.3f}\t'
          ' * Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    if is_best:
        torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def cal_t(t0, delt, iter):
    if t0+iter*delt>=1.0:
        return 1.0
    else:
        return t0+iter*delt

if __name__ == '__main__':
    main()

