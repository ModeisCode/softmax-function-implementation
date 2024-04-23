import torch
import numpy
from attentions import Attention

attention = Attention()

data = [1,2,5,7,9,3]
xi = torch.from_numpy(numpy.array(data))
print(xi)
softmax = attention.softmax(xi)
print(torch.softmax(xi,dim=0,dtype=float))
print("SOFTMAX:" , softmax)
