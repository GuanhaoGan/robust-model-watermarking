import torch
import torch.nn as nn

class CleanBatchNorm2d(nn.BatchNorm2d):
    """
    Modified from https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(CleanBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, mask=None):
        # mask==0 means clean sample, only use clean samples to calculate mean and variance
        
        # if mask is None:
        #     mask = torch.zeros(input.shape[0], dtype=torch.long).to(input.device)
        if mask is None or mask.sum().item()==0: # all clean, use BN to accelerate 
            return super().forward(input)
        self._check_input_dim(input)
        clean_num = input.shape[0]-mask.sum().item()

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative running average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential running average
                    exponential_average_factor = self.momentum

        # when training and there are clean samples, use clean samples to calculate statistics
        if self.training and clean_num: 
            
            clean_input = input[mask==0]
            mean = clean_input.mean([0, 2, 3])
            var = clean_input.var([0, 2, 3], unbiased=False)
            n = clean_input.numel() / clean_input.size(1)

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n-1)\
                    + (1 - exponential_average_factor) * self.running_var               
        elif not self.training:
            mean = self.running_mean
            var = self.running_var
        else:
            raise ValueError("No clean samples to compute clean statistics")

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            output = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        else:
            output = input
        return output
    

