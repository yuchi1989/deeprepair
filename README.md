# deeprepair


## [CIFAR-10](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/confusion) prototype  


## [COCO](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/coco ) prototype  



## Get Instance
### [coco get instance](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/get_instance.py)

```
python2 get_instance.py --original original_model/model_best.pth.tar --fix fix_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
python2 get_instance.py --original original_model/model_best.pth.tar --fix fix_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "bus" --second "person" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

### [coco gender get instance bias](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/get_instance_coco_gender_bias.py)

```
python2 get_instance_coco_gender_bias.py --original original_model/model_best.pth.tar --fix fix_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "skis" --second "man" --third "woman" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```


### [cifar10 get instance](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/cifar10_get_instance.py)

```
python3 cifar10_get_instance.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test  --cutmix_prob 0 --original ./runs/cifar10_resnet18_2_4/model_best.pth.tar --fix ./runs/cifar10_resnet_2_4_dogcat_dbr/model_best.pth.tar --expid 0  --checkmodel --first 5 --second 3

python3 cifar10_get_instance.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test  --cutmix_prob 0 --original ./runs/cifar10_resnet18_2_4/model_best.pth.tar --fix ./runs/cifar10_resnet_2_4_dogcat_dbr/model_best.pth.tar --expid 0  --checkmodel --first 3 --second 5

```

### [cifar100 get instance](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/cifar10_get_instance.py)

```
python3 cifar10_get_instance.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test  --cutmix_prob 0 --original ./runs/cifar100_resnet34/model_best.pth.tar --fix ./runs/cifar100_resnet34_oversampling/model_best.pth.tar --expid 0  --checkmodel --first 35 --second 98

python3 cifar10_get_instance.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test  --cutmix_prob 0 --original ./runs/cifar100_resnet34/model_best.pth.tar --fix ./runs/cifar100_resnet34_oversampling/model_best.pth.tar --expid 0  --checkmodel --first 98 --second 35

```



## [COCO](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments
### [COCO confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_exp_weighted_loss.py):  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 3 --class_num 80
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_bn.py):  

```
python2 repair_confusion_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair_bn --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.4
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode confusion --first "bus" --second "person"
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair_loss --first "bus" --second "person" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 80 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_dbr.py):  

```
python2 repair_confusion_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair_dbr --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --lam 0.5
```

### [COCO bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_exp_weighted_loss.py):  

```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_aug --first "bus" --second "person" --third "clock" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 3 --class_num 80
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_bn.py):  

```
python2 repair_bias_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_bn --first "bus" --second "person" --third "clock" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.4
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode bias --first "bus" --second "person" --third "clock"
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py):
```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_loss --first "bus" --second "person" --third "clock" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 80 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_dbr.py):  

```
python2 repair_bias_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_dbr --first "bus" --second "person" --third "clock" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --lam 0.5
```

## [COCO gender](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments
### [COCO gender confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_weighted_loss.py):  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair_aug --first "handbag" --second "woman" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 3 --class_num 81
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_bn.py):  

```
python2 repair_confusion_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair_bn --first "handbag" --second "woman" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.4
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode confusion --first "handbag" --second "woman"
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair_loss --first "handbag" --second "woman" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 81 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_dbr.py):  

```
python2 repair_confusion_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair_dbr --first "handbag" --second "woman" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --lam 0.5
```

### [COCO gender bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_exp_weighted_loss.py):  

```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_bias_repair_aug --first "handbag" --second "woman" --third "man" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 3 --class_num 81
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_bn.py):  

```
python2 repair_bias_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_bias_repair_bn --first "handbag" --second "woman" --third "man" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.4
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode bias --first "handbag" --second "woman" --third "man"
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py):
```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_bias_repair_loss --first "handbag" --second "woman" --third "man" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 81 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_dbr.py):  

```
python2 repair_bias_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_dbr --first "handbag" --second "woman" --third "man" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --lam 0.5
```  

### [COCO 2 pair confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_multipair) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/get_original_confusion.py):    

```
python2 get_original_confusion.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --num_epochs 15
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/repair_confusion_exp_weighted_loss.py):  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_2pair_confusion_repair_aug  --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 3 --class_num 80 --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --num_epochs 15
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/repair_confusion_bn.py):  

```
python2 repair_confusion_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_2pair_confusion_repair_bn  --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --ratio 0.4 --num_epochs 15
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode multipairconfusion --first person --second bus --third mouse --fourth keyboard
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_2pair_confusion_repair_loss --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 80 --target_weight 0.4 --num_epochs 15
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/repair_confusion_dbr.py):  

```
python2 repair_confusion_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_2pair_confusion_repair_dbr  --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --lam 0.5 --num_epochs 15
```

## [CIFAR-10](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10) experiments
### [CIFAR-10 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/confusion) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline.py):    

```
python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_resnet18_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel  
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
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --lam 0.5
```

### [CIFAR-10 bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/bias) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/train_baseline.py):    

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel  
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
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_dbr.py):  

```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --third 2 --lam 0.5
```


### [CIFAR-10 2 pair confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10_multipair/confusion) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/train_baseline.py):    

```
python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_resnet18_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_exp_oversampling.py):  

```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9  --weight 3.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_exp_newbn.py):  

```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9   --ratio 0.4  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_exp_newbn_softmax.py):
```
python3 repair_confusion_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9  
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_exp_weighted_loss.py):
```
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9   --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9 --lam 0.5
```




### [CIFAR-10 VGG-BN confusion](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline_vggbn.py) experiments
#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline.py):    

```
python3 train_baseline_vggbn.py  --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_vggbn_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
python3 repair_confusion_exp_newbn_vggbn.py --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel  
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_oversampling_vggbn.py):  

```
python3 repair_confusion_exp_oversampling_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_oversampling --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn_vggbn.py):  

```
python3 repair_confusion_exp_newbn_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_newbn --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --first 3 --second 5 --ratio 0.4  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn_softmax_vggbn.py):
```
python3 repair_confusion_exp_newbn_softmax_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_softmax --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.1 --checkmodel --first 3 --second 5
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss_vggbn.py):
```
python3 repair_confusion_exp_weighted_loss_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_loss --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_dbr_vggbn.py):  

```
python3 repair_confusion_dbr_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_vggbn_dbr --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --lam 0.5
```

### [CIFAR-10 VGG-BN bias](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline_vggbn.py) experiments

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn_vggbn.py):    

```
python3 repair_bias_exp_newbn_vggbn.py --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel   --first 3 --second 5 --third 2
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_bias_exp_oversampling_vggbn.py):  

```
python3 repair_bias_exp_oversampling_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_bias_vggbn_oversampling --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn_vggbn.py):  

```
python3 repair_bias_exp_newbn_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_bias_vggbn_newbn --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --first 3 --second 5 --third 2 --ratio 0.4  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn_softmax_vggbn.py):
```
python3 repair_bias_exp_newbn_softmax_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_bias_vggbn_softmax --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.1 --checkmodel --first 3 --second 5 --third 2
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss_vggbn.py):
```
python3 repair_bias_exp_weighted_loss_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_bias_vggbn_loss --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_dbr_vggbn.py):  

```
python3 repair_bias_dbr_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_bias_vggbn_dbr --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_vggbn_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --third 2 --lam 0.5
```





## [CIFAR-100 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100) experiments
### [CIFAR-100 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100/confusion) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/train_baseline_cifar100.py):    

```
python3 train_baseline_cifar100.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 256 --lr 0.1 --expname cifar100_resnet34 --epochs 300 --beta 1.0 --cutmix_prob 0
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar10_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel  
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_oversampling.py):  

```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 35 --second 98 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_newbn.py):  

```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --first 35 --second 98 --ratio 0.2  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_newbn_softmax.py):
```
python3 repair_confusion_exp_newbn_softmax.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.8 --checkmodel --first 35 --second 98
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_weighted_loss.py):
```
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 35 --second 98 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 256 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --first 35 --second 98 --lam 0.1
```

### [CIFAR-100 bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100/bias) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/train_baseline_cifar100.py):    

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel --first 98 --second 35 --third 11
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_oversampling.py):  

```
python3 repair_bias_exp_oversampling.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 98 --second 35 --third 11 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_newbn.py):  

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --ratio 0.4 --first 98 --second 35 --third 11
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_newbn_softmax.py):
```
python3 repair_bias_exp_newbn_softmax.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel --first 98 --second 35 --third 11
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_weighted_loss.py):
```
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 98 --second 35 --third 11 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_dbr.py):  

```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 256 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --first 98 --second 35 --third 11 --lam 0.1
```
