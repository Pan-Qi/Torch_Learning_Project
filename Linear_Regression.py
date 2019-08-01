import torch

w = torch.tensor(3.0, requires_grad = True)
b = torch.tensor(1.0, requires_grad = True)


def forward(x):
    y = w*x + b
    return y

x = torch.tensor([[4],[7]])
print("the calculation of LR perdiction:")
print(forward(x))
print("===================================")

from torch.nn import Linear

torch.manual_seed(1)
model = Linear(in_features=1, out_features=1)
print("bias & weight")
print(model.bias, model.weight)

x = torch.tensor([[2.0],[3.3]])
print("perdict by random parameters")
print(model(x))


import torch.nn as nn

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