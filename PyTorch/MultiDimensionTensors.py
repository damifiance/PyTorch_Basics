import torch

#0D Tensor(Scalar)

t0 = torch.tensor(1)
print(t0.ndim)
print(t0.shape)
print(t0)

#1D Tensor(Vector)

t1 = torch.tensor([1,2,3])
print(t1.ndim)
print(t1.shape)
print(t1)

#2D Tensor (Matrix)

t2 = torch.tensor([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print(t2.ndim)
print(t2.shape)
print(t2)

#3D Tensor (samples, timesteps, features)

t3 = torch.tensor([[[1,2,3],
                   [4,5,6],
                   [7,8,9]],
                   [[1,2,3],
                   [4,5,6],
                   [7,8,9]],
                   [[1,2,3],
                   [4,5,6],
                   [7,8,9]]])
print(t3.ndim)
print(t3.shape)
print(t3)

#4D Tensor (samples, height, width, channel)

#5D Tensor (samples, frames, height, width, channel)

