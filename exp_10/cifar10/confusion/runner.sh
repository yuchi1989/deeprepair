#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 train_baseline.py --net_type vgg --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_vgg_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0

CUDA_VISIBLE_DEVICES=1 python3 repair_confusion_exp_newbn.py --net_type vgg --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vgg_2_4_dogcat_test_newbn --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vgg_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel

CUDA_VISIBLE_DEVICES=1 python3 repair_confusion_exp_oversampling.py --net_type vgg --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vgg_2_4_dogcat_test_oversampling --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vgg_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --weight 2.0

CUDA_VISIBLE_DEVICES=1 python3 repair_confusion_exp_newbn.py --net_type vgg --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vgg_2_4_dogcat_test_newbn_2 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --first 3 --second 5 --ratio 0.2

CUDA_VISIBLE_DEVICES=1 python3 repair_confusion_exp_newbn_softmax.py --net_type vgg --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vgg_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vgg_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel --first 3 --second 5

CUDA_VISIBLE_DEVICES=1 python3 repair_confusion_exp_weighted_loss.py --net_type vgg --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vgg_2_4_dogcat_test_weighted_loss --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vgg_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --target_weight 0.4

CUDA_VISIBLE_DEVICES=1 python3 repair_confusion_dbr.py --net_type vgg --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname vgg_confusion_dbr --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vgg_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --lam 0.5
