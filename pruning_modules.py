import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np


class Conv_mask(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,bias):
        super(Conv_mask, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding)
        self.mask = nn.Parameter(torch.ones(out_channels))
        
    def forward(self, x):
        self.conv.weight.data = torch.einsum("cijk,c->cijk", self.conv.weight.data, self.mask)
        # print(self.conv.weight.data)
        return self.conv(x)
        
def structured_prune(pruneable_layers,prune_rate):
        convs = pruneable_layers    
        channel_norms = []
        for conv in convs:
            channel_norms.append(
                torch.sum(
                    torch.abs(conv.conv.weight.view(conv.conv.out_channels, -1)), axis=1
                )
            )
        values = []
        for x in channel_norms:
            for y in x.data.numpy():
                values.append(y)
        threshold = np.percentile(values, prune_rate)
        for conv in convs:
            channel_norms = torch.sum(
                torch.abs(conv.conv.weight.view(conv.conv.out_channels, -1)), axis=1
            )
            mask = conv.mask * (channel_norms < threshold)
            conv.conv.weight.data = torch.nn.parameter.Parameter(torch.einsum("cijk,c->cijk", conv.conv.weight.data, mask))
            
def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')

# class Conv_mask(nn.Module):
#     def __init__(
#         self,
#         in_planes,
#         planes,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         bias=False,
#     ):
#         super(Conv_mask, self).__init__()
#         self.conv = nn.Conv2d(
#             in_planes,
#             planes,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=bias,
#         )
#         self.mask = nn.Conv2d(
#             in_planes,
#             planes,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=bias,
#         )
#         self.weight = self.conv.weight
#         self.bias = self.conv.bias
#         self.device = self.conv.weight.device
#         self.mask.weight.data = torch.ones(self.mask.weight.size())
         

#     def forward(self, x):
#         self.weight.data = torch.mul(self.weight, self.mask.weight)
#         return self.conv(x)

#     def prune(self, threshold):
#         weight_dev = self.device
#         mask_dev = self.device
#         tensor = self.weight.data.cpu().numpy()
#         mask = self.mask.weight.data.cpu().numpy()
#         new_mask = np.where(abs(tensor) < threshold, 0, mask)
#         self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
#         self.mask.weight.data = torch.from_numpy(new_mask).to(mask_dev)
        
