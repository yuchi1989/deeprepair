# reproduce results on cifar10
# repair state-of-the-art cutmix model
# clone cutmix model
git clone https://github.com/clovaai/CutMix-PyTorch.git

# install the necessary environment from https://github.com/clovaai/CutMix-PyTorch
# see https://github.com/clovaai/CutMix-PyTorch

# copy train_baseline.py to cutmix folder and train baseline model
python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname cifar10_resnet_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
# check model cat-dog confusion
python3 cifar10_repair_confusion.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet_2_4/model_best.pth.tar --checkmodel

# repair model to reduce cat-dog confusion
python3 cifar10_repair_confusion.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet_2_4/model_best.pth.tar --expid 0 --lam 0.5
