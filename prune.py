import torch
import torch.nn as nn
import numpy as np
import pytorchyolo.train as train
import pytorchyolo.test as test
from pruning_modules import print_nonzeros
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.models import load_model
import argparse


parser = argparse.ArgumentParser(description="Get threshold.")
parser.add_argument("-sen", type=float, default=0.25, help="Sensitivity for pruning.")
parser.add_Argument("-train", type = bool, default = False, help = "Retrain Model.")
args = parser.parse_args()

print("Loading Model\n")
data = "config/coco.data"
model = "config/yolov3.cfg"
checkpoint_path = "weights/yolov3.weights"

data_config = parse_data_config(data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model,weights_path=checkpoint_path)
print_nonzeros(model)

print("Pruned Test Outcome\n")
checkpoint_path = model.prune_by_std(checkpoint_path= checkpoint_path,s = args.sen)
print_nonzeros(model)
test.run(model =model,weights=checkpoint_path)

if (args.train==True):
  print("Pruned and Retrained and Test Outcome\n")
  model, checkpoint_path = train.run(model=model,pretrained_weights= checkpoint_path)
  print_nonzeros(model)
  test.run(model=model,weights =checkpoint_path)


