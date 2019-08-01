import torch

x = torch.tensor(1.0, requires_grad = True)
z = torch.tensor(2.0, requires_grad = True)
y = x**2 + z**3
y.backward()

print("partial by x:" + str(x.grad))
print("partial by z:" + str(z.grad))