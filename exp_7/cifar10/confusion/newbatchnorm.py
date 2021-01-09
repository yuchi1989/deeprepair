import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import uniform
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter


class dnnrepair_BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, target_ratio=0, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(dnnrepair_BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.target_ratio = target_ratio

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            half_len = input.size(0)//2
            original_half = input[:half_len]
            target_half = input[half_len:]  # target input
            mean = original_half.mean(
                [0, 2, 3]) * (1 - self.target_ratio) + target_half.mean([0, 2, 3]) * self.target_ratio
            # use biased var in train
            var = original_half.var([0, 2, 3], unbiased=False) * (1 - self.target_ratio) + \
                target_half.var([0, 2, 3], unbiased=False) * self.target_ratio
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean.copy_(exponential_average_factor * mean +
                                        (1 - exponential_average_factor) * self.running_mean)
                # update running_var with unbiased var
                self.running_var.copy_(exponential_average_factor * var * n /
                                       (n - 1) + (1 - exponential_average_factor) * self.running_var)
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / \
            (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None,
                                        None] + self.bias[None, :, None, None]

        return input
