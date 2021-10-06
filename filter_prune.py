import cv2
from torch.nn import parameter
from pytorchyolo import detect, models
import time
import argparse
import numpy as np
import torch
from pruning_modules import print_nonzeros, structured_prune
import torch.nn as nn


def prune_network(model, channel_prune_perc, non_prune_channels,routed_to_channels,routed_from_channels):
    dim = 0
    routing_indices = []
    for m in range(len(model.module_list)):
        if model.module_list[m][0].__class__.__name__ == "Conv2d":
            if dim == 1:
                if 'conv_%d'%m in routed_to_channels:
                    model.module_list[m][0] = prune_module(model.module_list[m][0], "conv", 1, routing_indices[-1])

                elif 'conv_%d'%m == 'conv_87':
                    original_in_channels = model.module_list[m][0].weight.data.shape[1]
                    final_in_channels = conv_84 + conv_60
                    total_prunes = original_in_channels- final_in_channels
                    filter_sum_list = torch.sum(
                    torch.abs(model.module_list[m][0].weight.view(model.module_list[m][0].in_channels, -1)), axis=1)
                    vals, args = torch.sort(filter_sum_list)
                    channel_indices = args[:total_prunes].tolist()
                    model.module_list[m][0] = prune_module(model.module_list[m][0], "conv", 1, channel_indices)
            
                elif 'conv_%d'%m == 'conv_99':
                    original_in_channels = model.module_list[m][0].weight.data.shape[1]
                    final_in_channels = conv_96 + conv_35
                    total_prunes = original_in_channels- final_in_channels
                    filter_sum_list = torch.sum(
                    torch.abs(model.module_list[m][0].weight.view(model.module_list[m][0].in_channels, -1)), axis=1)
                    vals, args = torch.sort(filter_sum_list)
                    channel_indices = args[:total_prunes].tolist()
                    model.module_list[m][0] = prune_module(model.module_list[m][0], "conv", 1, channel_indices)
                else:
                    model.module_list[m][0] = prune_module(model.module_list[m][0], "conv", 1, channel_indices)
                    
                dim ^= 1
            
            if 'conv_%d'%m in non_prune_channels:
                dim = 1
                continue
            
            channel_indices = get_channel_indices(model.module_list[m][0], channel_prune_perc)
            model.module_list[m][0] = prune_module(model.module_list[m][0], "conv", 0, channel_indices)
            if (len(model.module_list[m]) > 1):
                model.module_list[m][1] = prune_module(model.module_list[m][1], "bn", 0, channel_indices)
            dim ^= 1
            if 'conv_%d'%m in routed_from_channels:
                routing_indices.append(channel_indices)
            if 'conv_%d'%m == 'conv_84':
                conv_84 = model.module_list[m][0].weight.data.shape[0]
            if 'conv_%d'%m == 'conv_60':
                conv_60 = model.module_list[m][0].weight.data.shape[0]
            if 'conv_%d'%m == 'conv_96':
                conv_96 = model.module_list[m][0].weight.data.shape[0]
            if 'conv_%d'%m == 'conv_35':
                conv_35 = model.module_list[m][0].weight.data.shape[0]
        else:
            continue 
        print(model.module_list[m])
    # model.module_list[m+1][0] = prune_module(model.module_list[m+1][0], "conv", 1, channel_indices)
    return model

def get_channel_indices(module, channel_prune_perc):
    out_channels = module.weight.data.shape[0]
    pruned_channels = int(channel_prune_perc * out_channels)
    filter_sum_list = torch.sum(
                torch.abs(module.weight.view(module.out_channels, -1)), axis=1)
    vals, args = torch.sort(filter_sum_list)
    return args[:pruned_channels].tolist()


def prune_module(module, module_type, dim, channel_indices):
    if len(channel_indices) == 0:
        return module
    if module_type == "conv":
        if dim == 0:
            new_conv = torch.nn.Conv2d(in_channels=module.in_channels,
                                    out_channels=int(module.out_channels - len(channel_indices)),
                                    kernel_size=module.kernel_size,
                                    stride=module.stride, padding=module.padding, dilation=module.dilation,bias = False)
            new_conv.weight.data = remove_indices(module.weight.data, dim, channel_indices)
            return new_conv

        elif dim == 1:
            new_conv = torch.nn.Conv2d(in_channels=int(module.in_channels - len(channel_indices)),
                                    out_channels=module.out_channels,
                                    kernel_size=module.kernel_size,
                                    stride=module.stride, padding=module.padding, dilation=module.dilation, bias = False)
            
            new_weight = remove_indices(module.weight.data,dim, channel_indices)
            new_conv.weight.data = new_weight
            return new_conv
        else:
            pass
    elif module_type == "bn":
        new_norm = torch.nn.BatchNorm2d(num_features=int(module.num_features - len(channel_indices)),
                                    eps=module.eps,
                                    momentum=module.momentum,
                                    affine=module.affine,
                                    track_running_stats=module.track_running_stats)
        if module.track_running_stats:
            new_norm.running_mean.data = remove_indices(module.running_mean.data, dim, channel_indices)
            new_norm.running_var.data = remove_indices(module.running_var.data, dim, channel_indices)
        return new_norm
    else:
        pass


def remove_indices(tensor, dim, channel_indices):
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(channel_indices)
    size_[dim] = new_size
    new_size = size_
    select_index = list(set(range(tensor.size(dim))) - set(channel_indices))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))  
    return new_tensor



model = models.load_model(
    "config/yolov3.cfg", 
    "weights/yolov3.weights",pruning=False)

channel_prune_perc = 0.99

non_prune_channels = ['conv_105', 'conv_93', 'conv_81']
routed_from_channels = ['conv_79', 'conv_91']
routed_to_channels = ['conv_84', 'conv_96']
upsample_route_channels = ['conv_87', 'conv_99']
model = prune_network(model, channel_prune_perc, non_prune_channels, routed_to_channels, routed_from_channels)

print(model)
torch.save(model, "custom_pruned_model.pth")
