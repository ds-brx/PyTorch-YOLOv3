import torch
import torch.nn as nn
import numpy as np
import pytorchyolo.train as train
import pytorchyolo.test as test
from pruning_modules import print_nonzeros
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.models import load_model
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prune Model.")
    parser.add_argument("-p", "--prune", type=bool, default=False, help="Prune model.")
    parser.add_argument("-s", "--sen", type = float, default = 0, help = "Add sensitivty.")
    parser.add_argument("-o", "--operation", type = str, default = "mean", help ="Operation to prune.")
    args = parser.parse_args()
    data = "config/coco.data"
    model = "config/yolov3.cfg"
    checkpoint_path = "weights/yolov3.weights"

    data_config = parse_data_config(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model,weights_path=checkpoint_path, pruning=False)
    print("Pruning by {}, with sensitivity {}.".format(args.operation, args.sen))
    for name, module in model.named_modules():
        if type(module).__name__ == "Conv2d":
            weights = module.weight.data.cpu().numpy()
            if args.operation == "mean":
                module_mean = np.mean(weights)
                threshold = abs(args.sen * module_mean)
            else:
                module_std = np.std(weights)
                threshold = abs(args.sen*module_std)
            new_weights = np.where(abs(weights) < threshold,0,weights)
            module.weight.data = torch.from_numpy(new_weights)
            print("Weights: {}".format(module.weight.data))

    print_nonzeros(model)
    test.run(model=model)
    model_path = "new_model.pth"
    torch.save(model.state_dict(), model_path)
