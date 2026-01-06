import torch
import numpy as np

a = torch.ones(7)
print(a)
b = a.numpy()
print(b)


a.add_(1)
print(a)
print(b) # CPU에서는 torch와 numpy는 메모리를 공유하기 때문에 b도 함께 변함


a = np.ones(7)
b = torch.from_numpy(a)
np.add(a, 1, out = a)
print(a)
print(b) # CPU에서는 torch와 numpy는 메모리를 공유하기 때문에 b도 함께 변함