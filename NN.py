import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn
from sklearn import datasets

n_pts = 500
x,y = datasets.make_circles(n_samples= n_pts, random_state=123, noise = 0.1, factor = 0.2)
x_data = torch.Tensor(x)
y_data = torch.Tensor(y.reshape(500,1))

def scatter_plot():
    plt.scatter(x[y==0,0],x[y==0,1])
    plt.scatter(x[y==1,0],x[y==1,1])
    plt.show()

scatter_plot()

class Model(nn.Module):
    def __init__(self, input_size, H1, H2, output_size):
        super().__init__ ()
        self.linear = nn.Linear(input_size, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, Houtput_size1)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred
    def predict(self,x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        return 0

torch.manual_seed(2)
model = Model(2,1)

[w,b] = model.parameters()
w1,w2 = w.view(2)

def get_params():
    return (w1.item(),w2.item(),b[0].item())

def plot_fit(title):
    plt.title = title
    w1,w2,b1 = get_params()
    x1 = np.array([-2.0,2.0])
    x2 = (w1*x1 + b1)/-w2
    plt.plot(x1,x2,'r')
    scatter_plot()

plot_fit('Init_Model')


criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

epochs = 1000
losses = []
for e in range(epochs):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred,y_data)
    print("epoch:", e, "loss:", loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plot_fit('Trained_Model')

plt.plot(range(epochs),losses)
plt.ylabel("Loss")
plt.xlabel('epoch')
plt.show()

print("prediction of point [1,5] is:")
print(model.predict(torch.Tensor([1,5])))