import torch
import torch.nn as nn
import numpy as np
import pytorchyolo.train as train
import pytorchyolo.test as test
from pruning_modules import print_nonzeros, structured_prune
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.models import load_model
import argparse


data = "config/coco.data"
model = "config/yolov3.cfg"
checkpoint_path = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model,weights_path=checkpoint_path, pruning=True)
print_nonzeros(model)

# model, checkpoint_path = train.run(model=model,pretrained_weights= None)
# print_nonzeros(model)
# test.run(model=model,weights =checkpoint_path)
print(model)


pruneable_layers = []
for name, module in model.named_modules():
    if module.__class__.__name__ == "Conv_mask":
        pruneable_layers.append(module)

structured_prune(pruneable_layers,5)
print_nonzeros(model)
torch.save(model.state_dict(), "new_model_pruned.pth")
