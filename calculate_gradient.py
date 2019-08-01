import torch

x = torch.tensor(2.0, requires_grad = True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x +1
y.backward()
print(x.grad)