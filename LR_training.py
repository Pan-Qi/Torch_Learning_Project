import torch
import torch.nn as nn
import matplotlib.pyplot as plt 

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
    
    def forward(self, x):
        pred = self.linear(x)
        return pred

torch.manual_seed(1)
model = LR(1,1)
print("parameters by class:")
print(list(model.parameters()))

x = torch.tensor([1.0])
print("perdict by class:")
print(model.forward(x))