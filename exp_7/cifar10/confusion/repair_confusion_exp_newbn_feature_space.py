# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
# python3 repair_retrain_exp.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --pretrained ./runs/DeepInspect_1/model_best.pth.tar --expid 0 --checkmodel
# python3 repair_retrain_exp.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname ResNet50 --epochs 60 --beta 1.0 --cutmix_prob 1.0 --pretrained ./runs/DeepInspect_1/model_best.pth.tar --expid 0 --first 3 --second 5

# python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 256
# set extra batch size same as batch size for half half assumption in new batchnorm layer
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
from tqdm import tqdm
from newbatchnorm2 import dnnrepair_BatchNorm2d
warnings.filterwarnings("ignore")

torch.manual_seed(124)
torch.cuda.manual_seed(124)
np.random.seed(124)
random.seed(124)
# torch.backends.cudnn.enabled=False
# torch.backends.cudnn.deterministic=True

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
parser.add_argument('--extra', default=10, type=int,
                    help='extra batch size')
parser.add_argument('--keeplr', help='set lr 0.001 ',
                    action='store_true')

parser.add_argument('--replace', help='replace bn layer ',
                    action='store_true')

parser.add_argument('--ratio', default=0.5, type=float,
                    help='target ratio for batchnorm layers')

parser.add_argument('--groupname', default="", type=str, help='experiment id')
# parser.add_argument('--forward', default=1, type=int,
#                    help='extra batch size')
parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=False)

best_loss = 100
best_err1 = 100
best_err5 = 100
global_epoch_confusion = []


def log_print(var):
    print("logging filter: " + str(var))


glob_bn_total = 0
glob_bn_count = 0


def count_bn_layer(module):
    global glob_bn_total
    for child_name, child in module.named_children():
        if isinstance(child, torch.nn.modules.batchnorm.BatchNorm2d):
            #setattr(module, child_name, nn.Softplus())
            glob_bn_total += 1
        else:
            count_bn_layer(child)


def replace_bn(module):
    global glob_bn_count
    global glob_bn_total
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present

    for child_name, child in module.named_children():
        if isinstance(child, torch.nn.modules.batchnorm.BatchNorm2d):
            glob_bn_count += 1
            if glob_bn_count >= glob_bn_total - 2:  # unfreeze last 3
                print('replaced: bn')
                #new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 0.5, child.eps, child.momentum, child.affine, track_running_stats=True)
                #new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 9/19, child.eps, 0.19, child.affine, track_running_stats=True)
                #new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 9/19, child.eps, child.momentum, child.affine, track_running_stats=True)
                new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, args.ratio, child.eps, child.momentum, child.affine, track_running_stats=True)
                setattr(module, child_name, new_bn)
            else:
                print('replaced: bn')
                new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 0, child.eps, child.momentum, child.affine, track_running_stats=True)
                setattr(module, child_name, new_bn)
        else:
            replace_bn(child)

def set_bn_eval(model):
    global glob_bn_count
    global glob_bn_total
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            glob_bn_count += 1
            if glob_bn_count < glob_bn_total - 2:  # unfreeze last 3
                # if glob_bn_count < glob_bn_total:# unfreeze last bn
                # if glob_bn_count != glob_bn_total//2:# unfreeze middle bn
                # if glob_bn_count != 1: # unfreeze first bn layer
                # if glob_bn_count < glob_bn_total*2/3:# unfreeze last 1/3
                # if glob_bn_count > glob_bn_total*1/3:# unfreeze first 1/3
                # if glob_bn_count > glob_bn_total*2/3 or glob_bn_count < glob_bn_total*1/3: # unfreeze middle 1/3
                module.eval()
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
            else:
                module.momentum = 0.5


def set_bn_train(model):  # unfreeze all bn
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            #print("set bn")
            module.train()


def get_dataset_from_specific_classes(target_dataset, first, second):
    first_indices = np.where(np.array(target_dataset.targets) == first)[0]
    second_indices = np.where(np.array(target_dataset.targets) == second)[0]
    target_idx = np.hstack([first_indices, second_indices])
    target_dataset.targets = np.array(target_dataset.targets)[target_idx]
    target_dataset.data = target_dataset.data[target_idx]
    return target_dataset


layer_outputs= []
def hook(module, input, output):
    layer_outputs += output.detach().cpu().numpy()

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

            target_train_dataset = datasets.CIFAR10(
                '../data', train=True, download=True, transform=transform_train)
            target_train_dataset = get_dataset_from_specific_classes(
                target_train_dataset, args.first, args.second)
            target_test_dataset = datasets.CIFAR10(
                '../data', train=False, download=True, transform=transform_test)
            target_test_dataset = get_dataset_from_specific_classes(
                target_test_dataset, args.first, args.second)
            target_train_loader = torch.utils.data.DataLoader(target_train_dataset, batch_size=args.extra, shuffle=True,
                                                              num_workers=args.workers, pin_memory=True)
            target_val_loader = torch.utils.data.DataLoader(target_test_dataset, batch_size=args.extra, shuffle=True,
                                                            num_workers=args.workers, pin_memory=True)

        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))
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

    # print(model)
    print('the number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # replace bn layer
    if args.replace:
        model.to('cpu')
        global glob_bn_count
        global glob_bn_total
        glob_bn_total = 0
        glob_bn_count = 0
        count_bn_layer(model)
        print("total bn layer: " + str(glob_bn_total))
        glob_bn_count = 0
        replace_bn(model)
        print(model)
        model = model.cuda()

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
        repaired_model = 'runs/%s/' % (args.expname) + 'model_best.pth.tar'
        if os.path.isfile(repaired_model):
            print("=> loading checkpoint '{}'".format(repaired_model))
            checkpoint = torch.load(repaired_model)
            model.load_state_dict(checkpoint['state_dict'])

            model.module.avgpool.register_forward_hook(hook)
            labels = get_confusion(val_loader, model, criterion)
            # dog->cat confusion
            log_print(str(args.first) + " -> " + str(args.second))
            log_print(global_epoch_confusion[-1]
                    ["confusion"][(args.first, args.second)])
            # cat->dog confusion
            log_print(str(args.second) + " -> " + str(args.first))
            log_print(global_epoch_confusion[-1]
                    ["confusion"][(args.second, args.first)])
        outputs = np.array(layer_outputs)
        print(outputs.shape)
        labels = np.array(labels)
        print(labels.shape)
        np.save(args.groupname + '_outputs.npy', outputs)
        np.save(args.groupname + '_labels.npy', labels)

   




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

        output = model(input)
        _, top1_output = output.max(1)
        total += target.size(0)
        correct += top1_output.eq(target).sum().item()

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.mean().item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

        for i in range(len(input)):
            yhats.append(int(top1_output[i].cpu().data.numpy()))
            labels.append(int(target[i].cpu().data.numpy()))
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    acc = 100.*correct/total
    log_print(acc)

    correct = 0
    for i in range(len(labels)):
        if labels[i] == yhats[i]:
            correct += 1
    log_print(correct*1.0/len(labels))

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
    global_epoch_confusion[-1]["confusion"] = type1confusion
    global_epoch_confusion[-1]["accuracy"] = acc

    dog_cat_sum = 0
    dog_cat_acc = 0
    for i in range(len(yhats)):

        if args.first == labels[i] or args.second == labels[i]:
            dog_cat_sum += 1
            if labels[i] == yhats[i]:
                dog_cat_acc += 1
    global_epoch_confusion[-1]["dogcatacc"] = dog_cat_acc/dog_cat_sum
    log_print("pair accuracy: " + str(global_epoch_confusion[-1]["dogcatacc"]))

    return labels


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
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
