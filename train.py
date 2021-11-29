from pytorchyolo import train, test
import os
import argparse
import torch

if __name__ == "__main__":

    if not os.path.isdir('trained_pruned_models'):
        os.makedirs('trained_pruned_models')

    parser = argparse.ArgumentParser(description="train pruned model")
    parser.add_argument("-m", "--model", type=str, default="custom_pruned_models/cluster_prune_10.pth", help="Path to model")
    parser.add_argument("-e", '--epochs', type = int, default=100, help='epochs')
    args = parser.parse_args()

    model = torch.load(args.model)
    model = train.run(model, epochs=args.epochs)


    test.run(model)

    path_name = args.model.split('/')[-1].split('.')[0]
    torch.save(model, 'trained_pruned_models/{}_{}.pth'.format(path_name, args.epochs))

