# deeprepair


### [CIFAR-10](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/confusion) prototype  


### [COCO](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/coco ) prototype  
 








## Paper experiments
### [COCO](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments


### [COCO gender](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments
  

### [CIFAR-10 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/confusion) experiments  
  
#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline.py):    

```
python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_resnet18_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --checkmodel  
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_oversampling.py):  
  
```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn.py):  
  
```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --first 3 --second 5 --ratio 0.2  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn_softmax.py): 
```
python3 repair_confusion_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel --first 3 --second 5
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py): 
```
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_dbr.py):  
  
```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5
```

### [CIFAR-10 bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/bias) experiments  
  
#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/train_baseline.py):    

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --checkmodel  
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_oversampling.py):  
  
```
python3 repair_bias_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn.py):  
  
```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --ratio 0.2 --first 3 --second 5 --third 2
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn_softmax.py): 
```
python3 repair_bias_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel --first 3 --second 5 --third 2
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py): 
```
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_dbr.py):  
  
```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --third 2
```

### [CIFAR-100](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/cifar100/confusion) experiments  

### [CIFAR-100 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100/confusion) experiments  
  
#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/train_baseline.py):    

```
python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_resnet18_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --checkmodel  
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_oversampling.py):  
  
```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_newbn.py):  
  
```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --first 3 --second 5 --ratio 0.2  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_newbn_softmax.py): 
```
python3 repair_confusion_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel --first 3 --second 5
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_weighted_loss.py): 
```
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_dbr.py):  
  
```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5
```

### [CIFAR-100 bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100/bias) experiments  
  
#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/train_baseline.py):    

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --checkmodel  
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_oversampling.py):  
  
```
python3 repair_bias_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_newbn.py):  
  
```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --ratio 0.2 --first 3 --second 5 --third 2
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_newbn_softmax.py): 
```
python3 repair_bias_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel --first 3 --second 5 --third 2
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_weighted_loss.py): 
```
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_dbr.py):  
  
```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --third 2
```



