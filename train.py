from pytorchyolo.train import run
import os
import argparse
import torch

if not os.path.isdir('trained_pruned_models'):
    os.makedirs('trained_pruned_models')

parser = argparse.ArgumentParser(description="train pruned model")
parser.add_argument("-m", "--model", type=str, default="custom_pruned_models", help="Path to model")
args = parser.parse_args()

model = torch.load(args.model)
model = run(model)



