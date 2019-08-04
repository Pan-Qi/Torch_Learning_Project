import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn
from sklearn import datasets

n_pts = 100
centers = [[-0.5,0.5],[0.5,-0.5]]
x,y = datasets.make_blobs(n_samples= n_pts, random_state=123,
centers = centers, cluster_std = 0.4)
x_data = torch.tensor(x)
y_data = torch.tensor(y)

def scatter_plot():
    plt.scatter(x[y==0,0],x[y==0,1])
    plt.scatter(x[y==1,0],x[y==1,1])
    plt.show()

scatter_plot()

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__ ()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred

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