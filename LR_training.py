import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np

X = torch.randn(100,1)*10
y = X*5 + torch.randn(100,1)*7 + 26
# plt.plot(X.numpy(), y.numpy(), 'o')
# plt.ylabel('y')
# plt.xlabel('x')
# plt.show()

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
    
    def forward(self, x):
        pred = self.linear(x)
        return pred

print("===============")
torch.manual_seed(1)
model = LR(1,1)
print("parameters by class:")
print(list(model.parameters()))


[w,b] = model.parameters()
def get_params():
    return (w[0][0].item(), b[0].item())

def plot_fit(title):
    plt.title = title
    w1,b1 = get_params()
    x1 = np.array([-30,30])
    y1 = w1*x1 + b1
    plt.plot(x1,y1,'r')
    plt.scatter(X,y)
    plt.show()

plot_fit("Init")


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred,y)
    print("epoch:", i, "loss:", loss.item())


    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plot_fit("final")

plt.plot(range(epochs),losses)
plt.ylabel("Loss")
plt.xlabel('epoch')
plt.show()