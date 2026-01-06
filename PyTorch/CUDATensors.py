import torch
x = torch.randn(1)

print(x)
print(x.item())
print(x.dtype)

device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
print(device)

y = torch.ones_like(x, device=device)
print(x)

x = x.to(device)

z = x + y
print(z)
print(z.to('cpu', torch.double))

#TS only available when my laptop has a GPU