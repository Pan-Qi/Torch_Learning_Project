import torch
import numpy as np
# import numpy as np

v = torch.tensor([1,2,3,4,5,6])
print(v)
print(v.dtype)


f = torch.FloatTensor([1,2,3])
print(f)
print(f.dtype)
print(f.size())

print("===========================")
a = np.array([1,2,3,4,5])
tensor_cnv = torch.from_numpy(a)
print("convert np array to torch tensor")
print(tensor_cnv)
print(tensor_cnv.type())

print("convert torch tensor to np array")
numpy_cnv = tensor_cnv.numpy()
print(numpy_cnv)