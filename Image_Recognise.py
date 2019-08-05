import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import numpy as np 

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])

training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
training_dataset

