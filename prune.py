import torch
import torch.nn as nn
import numpy as np
import pytorchyolo.train as train
import pytorchyolo.test as test
from pruning_modules import print_nonzeros

##inital training
print("Initial Training\n")
model = 'config/yolov3.cfg'
model, checkpoint_path= train.run(model = model)
print_nonzeros(model)
test.run(model,checkpoint_path)

##Prune and test
print("Pruned Test Outcome\n")
checkpoint_path = model.prune_by_std()
print_nonzeros(model)
test.run(model =model,weights=checkpoint_path)


## Prune, Retrain and test
print("Pruned and Retrained and Test Outcome\n")
model, checkpoint_path = train.run(model=model)
print_nonzeros(model)
test.run(model=model,weights =checkpoint_path)


