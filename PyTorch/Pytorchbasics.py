import torch

x = torch.empty(4,2)
print(x)

x = torch.rand(4,2)
print(x)

x = torch.zeros(4,2)
print(x)

x= torch.zeros(4,2, dtype = torch.long)
print(x)

x= torch.tensor([3,2.3])
print(x)

x = x.new_ones(2,4, dtype= torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

ft = torch.FloatTensor([1,2,3])
print(ft)
print(ft.short())
print(ft.int())
print(ft.long())

it = torch.IntTensor([1,2,3])
print(it, it.dtype)

print(it.float())
print(it.double())
print(it.half())