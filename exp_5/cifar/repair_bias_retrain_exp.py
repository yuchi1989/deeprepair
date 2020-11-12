# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
# python3 repair_retrain_exp.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --pretrained ./runs/DeepInspect_1/model_best.pth.tar --expid 0 --checkmodel
# python3 repair_retrain_exp.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname ResNet50 --epochs 60 --beta 1.0 --cutmix_prob 1.0 --pretrained ./runs/DeepInspect_1/model_best.pth.tar --expid 0 --first 3 --second 5

import argparse
import os
import shutil
import time
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np
import random
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

torch.manual_seed(124)
torch.cuda.manual_seed(124)
np.random.seed(124)
random.seed(124)
#torch.backends.cudnn.enabled=False
#torch.backends.cudnn.deterministic=True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument(
    '--pretrained', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--expid', default="0", type=str, help='experiment id')
parser.add_argument('--checkmodel', help='Check model accuracy',
    action='store_true')
parser.add_argument('--lam', default=0.5, type=float,
                    help='hyperparameter lambda')
parser.add_argument('--first', default=3, type=int,
                    help='first object index')
parser.add_argument('--second', default=5, type=int,
                    help='second object index')
parser.add_argument('--third', default=5, type=int,
                    help='third object index')
parser.add_argument('--keeplr', help='set lr 0.001 ',
    action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_loss = 100
best_err1 = 100
best_err5 = 100
global_epoch_confusion = []


def main():
    global args, best_err1, best_err5, global_epoch_confusion, best_loss
    args = parser.parse_args()

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=True,
                                  download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False,
                                  transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True,
                                 download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False,
                                 transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join('/home/data/ILSVRC/train')
        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth,
                          numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception(
            'unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.pretrained))

    print(model)
    print('the number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True
    #validate(val_loader, model, criterion, 0)

    # for checking pre-trained model accuracy and confusion
    if args.checkmodel:
        global_epoch_confusion.append({})
        get_confusion(val_loader, model, criterion)
        confusion_matrix = global_epoch_confusion[-1]["confusion"]
        print("loss: " + str(global_epoch_confusion[-1]["loss"]))
        bias_matrix = global_epoch_confusion[-1]["bias"]
        print("loss: " + str(global_epoch_confusion[-1]["loss"]))
        print(str((args.second, args.third)) + " bias: " + 
            str(bias_matrix[(args.second, args.third)]))
        exit()

    for epoch in range(0, args.epochs):
        global_epoch_confusion.append({})
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        
        is_best = val_loss <= best_loss
        best_loss = min(val_loss, best_loss)
        if is_best:
            best_err5 = err5
            best_err1 = err1

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)


        get_confusion(val_loader, model, criterion, epoch)
        confusion_matrix = global_epoch_confusion[-1]["confusion"]
        print("loss: " + str(global_epoch_confusion[-1]["loss"]))
        bias_matrix = global_epoch_confusion[-1]["bias"]
        print("loss: " + str(global_epoch_confusion[-1]["loss"]))
        print(str((args.second, args.third)) + " bias: " + 
            str(bias_matrix[(args.second, args.third)]))


    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    epoch_confusions = 'runs/%s/' % (args.expname) + \
        'epoch_confusion_' + args.expid
    np.save(epoch_confusions, global_epoch_confusion)

    # output best model accuracy and confusion
    repaired_model = 'runs/%s/' % (args.expname) + 'model_best.pth.tar'
    if os.path.isfile(repaired_model):
        print("=> loading checkpoint '{}'".format(repaired_model))
        checkpoint = torch.load(repaired_model)
        model.load_state_dict(checkpoint['state_dict'])
        get_confusion(val_loader, model, criterion)
        bias_matrix = global_epoch_confusion[-1]["bias"]
        print("loss: " + str(global_epoch_confusion[-1]["loss"]))
        print(str((args.second, args.third)) + " bias: " + 
            str(bias_matrix[(args.second, args.third)]))

def compute_confusion(confusion_matrix, first, second):
    confusion = 0
    if (first, second) in confusion_matrix:
        confusion += confusion_matrix[(first, second)]
    
    if (second, first) in confusion_matrix:
        confusion += confusion_matrix[(second, first)]
    return confusion/2

def compute_bias(confusion_matrix, first, second, third):
    return abs(compute_confusion(confusion_matrix, first, second) - compute_confusion(confusion_matrix, first, third))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        target_copy = target.cpu().numpy()
        output2 = model(input)
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample

            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index,
                                                      :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                       (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a).mean() * lam + \
                criterion(output, target_b).mean() * (1. - lam)
            id3 = []
            id5 = []
            id1 = []
            for j in range(len(input)):
                if (target_copy[j]) == args.first:
                    id3.append(j)
                elif (target_copy[j]) == args.second:
                    id5.append(j)
                elif (target_copy[j]) == args.third:
                    id1.append(j)

            m = nn.Softmax(dim=1)

            p_dist1 = torch.dist(torch.mean(
                m(output2)[id3], 0), torch.mean(m(output2)[id5], 0), 2)
            p_dist2 = torch.dist(torch.mean(
                m(output2)[id3], 0), torch.mean(m(output2)[id1], 0), 2)
            loss2 = loss + args.lam * torch.square(p_dist1 - p_dist2)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)
            #_, top1_output = output.max(1)
            #yhats = top1_output.cpu().data.numpy()
            # print(yhats[:5])
            id3 = []
            id5 = []
            id1 = []

            filered_target = [id for id in target_copy if id != args.second]
            filered_target = [id for id in filered_target if id != args.third]
            first = Counter(filered_target).most_common(1)[0][0] 
            for j in range(len(input)):
                if (target_copy[j]) == first:
                    id3.append(j)
                elif (target_copy[j]) == args.second:
                    id5.append(j)
                elif (target_copy[j]) == args.third:
                    id1.append(j)
            m = nn.Softmax(dim=1)
            if len(id5) == 0 or len(id1) == 0:
                diff_dist = 0
                p_dist1 = 0
                p_dist2 = 0
            else:
                p_dist1 = torch.dist(torch.mean(
                    m(output)[id3], 0), torch.mean(m(output)[id5], 0), 2)
                p_dist2 = torch.dist(torch.mean(
                    m(output)[id3], 0), torch.mean(m(output)[id1], 0), 2)
                diff_dist = torch.abs(p_dist1 - p_dist2)
            #loss2 = loss.mean() + args.lam * diff_dist
            loss2 = loss.mean() - args.lam * (p_dist1  + p_dist2)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss2.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        '''
        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
        '''
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        target_copy = target.cpu().numpy()
        output = model(input)
        loss = criterion(output, target)
        id3 = []
        id5 = []
        id1 = []
        filered_target = [id for id in target_copy if id != args.second]
        filered_target = [id for id in filered_target if id != args.third]
        first = Counter(filered_target).most_common(1)[0][0] 

        for j in range(len(input)):
            if (target_copy[j]) == first:
                id3.append(j)
            elif (target_copy[j]) == args.second:
                id5.append(j)
            elif (target_copy[j]) == args.third:
                id1.append(j)
        m = nn.Softmax(dim=1)
        if len(id5) == 0 or len(id1) == 0:
            diff_dist = 0
            p_dist1 = 0
            p_dist2 = 0
        else:
            p_dist1 = torch.dist(torch.mean(
                m(output)[id3], 0), torch.mean(m(output)[id5], 0), 2)
            p_dist2 = torch.dist(torch.mean(
                m(output)[id3], 0), torch.mean(m(output)[id1], 0), 2)
            diff_dist = torch.abs(p_dist1 - p_dist2)
        #loss2 = loss.mean() + args.lam * diff_dist
        loss2 = loss.mean() - args.lam * (p_dist1  + p_dist2)
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss2.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        '''
        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))
        '''
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def get_confusion(val_loader, model, criterion, epoch=-1):
    global global_epoch_confusion
    global_epoch_confusion[-1]["epoch"] = epoch
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct = 0
    total = 0
    yhats = []
    labels = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        target_copy = target.cpu().numpy()
        output = model(input)
        _, top1_output = output.max(1)
        total += target.size(0)
        correct += top1_output.eq(target).sum().item()

        loss = criterion(output, target)
        id3 = []
        id5 = []
        id1 = []

        filered_target = [id for id in target_copy if id != args.second]
        filered_target = [id for id in filered_target if id != args.third]
        first = Counter(filered_target).most_common(1)[0][0] 

        for j in range(len(input)):
            if (target_copy[j]) == first:
                id3.append(j)
            elif (target_copy[j]) == args.second:
                id5.append(j)
            elif (target_copy[j]) == args.third:
                id1.append(j)
        m = nn.Softmax(dim=1)
        if len(id5) == 0 or len(id1) == 0:
            diff_dist = 0
            p_dist1 = 0
            p_dist2 = 0
        else:
            p_dist1 = torch.dist(torch.mean(
                m(output)[id3], 0), torch.mean(m(output)[id5], 0), 2)
            p_dist2 = torch.dist(torch.mean(
                m(output)[id3], 0), torch.mean(m(output)[id1], 0), 2)
            diff_dist = torch.abs(p_dist1 - p_dist2)
        #loss2 = loss.mean() + args.lam * diff_dist
        loss2 = loss.mean() - args.lam * (p_dist1  + p_dist2)
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss2.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        '''
        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))
        '''
        for i in range(len(input)):
            yhats.append(int(top1_output[i].cpu().data.numpy()))
            labels.append(int(target[i].cpu().data.numpy()))
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    acc = 100.*correct/total
    print(acc)

    correct = 0
    for i in range(len(labels)):
        if labels[i] == yhats[i]:
            correct += 1
    print(correct*1.0/len(labels))

    labels_list = []
    for i in range(10):
        labels_list.append(i)

    type1confusion = {}

    for l1 in labels_list:
        for l2 in labels_list:
            if l1 == l2 or ((l1, l2) in type1confusion):
                continue
            c = 0
            subcount = 0
            for i in range(len(yhats)):

                if l1 == labels[i] and l2 == yhats[i]:
                    c = c + 1

                if l1 == labels[i]:
                    subcount = subcount + 1

            type1confusion[(l1, l2)] = c*1.0/subcount
    avg_pair_confusion = {}
    bias = {}
    for i in range(10):
        for j in range(i + 1, 10):
            avg_pair_confusion[(i,j)] = (type1confusion[(i, j)] + type1confusion[(j, i)])/2
            avg_pair_confusion[(j,i)] = (type1confusion[(i, j)] + type1confusion[(j, i)])/2

    triplet_bias = {}
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i == j or j == k or i ==k:
                    continue
                triplet_bias[(i, j, k)] = abs(avg_pair_confusion[(i, j)] - avg_pair_confusion[(i, k)])

    pairwise_bias = {}
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            pairwise_bias[(i, j)] = 0
            for k in range(10):
                if j == k or i ==k:
                    continue
                pairwise_bias[(i, j)] += triplet_bias[(k, i, j)]

    global_epoch_confusion[-1]["confusion"] = type1confusion
    global_epoch_confusion[-1]["bias"] = pairwise_bias
    global_epoch_confusion[-1]["accuracy"] = acc
    global_epoch_confusion[-1]["loss"] = losses.avg

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        print("saving best model...")
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) +
                         'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global global_epoch_confusion
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * \
            (0.1 ** (epoch // (args.epochs * 0.75)))
        if args.keeplr:
            lr = 0.001
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    global_epoch_confusion[-1]["lr"] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()