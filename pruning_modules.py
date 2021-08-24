import torch
import torch.nn as nn
import numpy as np

class PruningModule(nn.Module):
    def prune_by_percentile(self, q=5.0, **kwargs):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        # Calculate percentile value
        alive_parameters = []
        for name, p in self.named_parameters():
            # We do not prune bias termx
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
            alive_parameters.append(alive)

        all_alives = np.concatenate(alive_parameters)
        percentile_value = np.percentile(abs(all_alives), q)
        print(f'Pruning with threshold : {percentile_value}')
    
    def prune_by_std(self, checkpoint_path, s=0.25):
        """
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        for name, module in self.named_modules():
            if 'Conv' in name:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)
        torch.save(self.state_dict(), checkpoint_path)
        return checkpoint_path



class UnstructuredMask:
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias=None):
        self.mask = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.mask.weight.data = torch.ones(self.mask.weight.size())

    def apply(self, conv):
        conv.weight.data = torch.mul(conv.weight, self.mask.weight)
    def get_weights_mask(self):
        return self.mask.weight.data.cpu().numpy()
    def assign_new_mask(self, mask_dev, new_mask):
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)


class StructuredMask:
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias=None):
        self.mask = nn.Parameter(torch.ones(planes))

    def apply(self, conv, bn):
        conv.weight.data = torch.einsum("cijk,c->cijk", conv.weight.data, self.mask)


class Conv_mask(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    ):
        super(Conv_mask, self).__init__()
        UnstructuredMask.__init__(self, in_planes, planes, kernel_size, stride, padding, bias=None)
        self.conv = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.mask = UnstructuredMask(
            in_planes, planes, kernel_size, stride, padding, bias
        )

    def forward(self, x):
        self.mask.apply(self.conv)

        return self.conv(x)

    def prune(self, threshold):
        weight_dev = self.conv.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.get_weights_mask()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.conv.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.assign_new_mask(mask_dev, new_mask)
        


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
