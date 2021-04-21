# deeprepair


### [CIFAR-10](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/confusion) prototype  


### [COCO](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/coco ) prototype  









## Paper experiments
### [COCO](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments
### [COCO confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_oversampling.py):  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 2 --class_num 80
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_bn.py):  

```
python2 repair_confusion_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.5
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode confusion
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair --first "bus" --second "person" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 80 --target_weight 0.5
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_dbr.py):  

```
python2 repair_confusion_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "person" --second "bus"
```

### [COCO bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_oversampling.py):  

```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "bus" --second "person" --third "clock" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 2 --class_num 80
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_bn.py):  

```
python2 repair_bias_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair --first "bus" --second "person" --third "clock" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.5
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode bias
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py):
```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair --first "bus" --second "person" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 80 --target_weight 0.5
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_dbr.py):  

```
python2 repair_bias_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair --first "bus" --second "person" --third "clock"
```

### [COCO gender](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments
### [COCO gender confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_oversampling.py):  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair --first "bus" --second "person" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 2 --class_num 81
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_bn.py):  

```
python2 repair_confusion_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair --first "woman" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.5
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode confusion
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair --first "bus" --second "person" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 81 --target_weight 0.5
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_dbr.py):  

```
python2 repair_confusion_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair --first "woman" --second "bus"
```

### [COCO gender bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_oversampling.py):  

```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_bias_repair --first "bowl" --second "woman" --third "man" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 2 --class_num 81
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_bn.py):  

```
python2 repair_bias_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_bias_repair --first "bowl" --second "woman" --third "man" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.5
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/home/yuchi/data/coco/annotations' --image_dir '/home/yuchi/data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode bias
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py):
```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair --first "bus" --second "person" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 81 --target_weight 0.5
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_dbr.py):  

```
python2 repair_bias_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair --first "bowl" --second "woman" --third "man"
```  

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
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5
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
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_dbr.py):  

```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --third 2
```



### [CIFAR-100 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100/confusion) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/train_baseline_cifar100.py):    

```
python3 train_baseline_cifar100.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 256 --lr 0.1 --expname cifar100_resnet34 --epochs 300 --beta 1.0 --cutmix_prob 0

# top confusion pairs
python3 analyze_cifar100_epoch_confusion.py --npy "../runs/cifar100_resnet34/epoch_confusion.npy" 

python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar10_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel --first 35 --second 98
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
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 35 --second 98
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 256 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --first 35 --second 98
```

### [CIFAR-100 bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100/bias) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/train_baseline_cifar100.py):    

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel  
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_oversampling.py):  

```
python3 repair_bias_exp_oversampling.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 35 --second 98 --third 2 --weight 2.0
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_newbn.py):  

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --ratio 0.2 --first 35 --second 98 --third 2
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_newbn_softmax.py):
```
python3 repair_bias_exp_newbn_softmax.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel --first 35 --second 98 --third 2
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_weighted_loss.py):
```
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 35 --second 98 --third 2
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_dbr.py):  

```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 256 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar100_resnet34/model_best.pth.tar --expid 0 --first 35 --second 98 --third 2
```
