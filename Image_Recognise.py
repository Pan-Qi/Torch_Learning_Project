import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import numpy as np 
from torch import nn
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))
                                ])

training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
training_dataset

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)


def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1,2,0)
    image = image*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    image = image.clip(0,1)
    return image

# print(training_dataset)
# print(training_dataset.clone())
# print(training_dataset.detach())


dataiter = iter(training_loader)
images, labels = dataiter.next()
fig = plt.figure(figsize=(25,4))

for idx in np.arange(20):
    ax = fig.add_subplot(2,10,idx+1,xticks=[],yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title([labels[idx].item()])

plt.show()


class Classifier(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in,H1)
        self.linear2 = nn.Linear(H1,H2)
        self.linear3 = nn.Linear(H2,D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
model = Classifier(784,125,65,10)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 12
losses = []
accuracy = []

for e in range(epochs):
    #1,28,28 to 784

    running_loss = 0.0
    running_corrects = 0.0
    for inputs,labels in training_loader:
        inputs = inputs.view(inputs.shape[0],-1)
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _,preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    else:
        epoch_loss = running_loss/len(training_loader)
        epoch_acc = running_corrects.float()/len(training_loader)
        losses.append(epoch_loss)
        accuracy.append(epoch_acc)
        print('trianing loss: {:.4f},{:.4f}'.format(epoch_loss,epoch_acc.item()))
    

plt.plot(losses,label='training loss')
plt.show()

plt.plot(accuracy,label='accuracy')
plt.show()