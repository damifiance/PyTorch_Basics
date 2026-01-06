import torch
import math

a = torch.rand(1,2) * 2 - 1
print(a)
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

print(a)
print(torch.min(a))
print(torch.max(a))
print(torch.mean(a))
print(torch.std(a))
print(torch.prod(a))

print(torch.unique(torch.tensor([1,2,3,1,2,2])))

x = torch.rand(2,2)
print(x)
print(x.max(dim=0)) #열에서 비교해줌
print(x.max(dim=1)) #행에서 비교해줌

x= torch.rand(2,2)
y = torch.rand(2,2)

#torch.add

print(x + y)
print(torch.add(x,y))

result = torch.empt(2,2)
torch.add(x,y, out = result)
print(result)

print(x)
print(y)
y.add_(x) # y += x랑 같은 뜻 (y에 y+x 저장) #이런 식의 _ 이용한 저장을 In place 라고 부름
print(y)

print(x-y)
print((torch.sub(x,y)))
print(x.sub(y))

print(x*y)
print(torch.mul(x,y))
print(x.mul(y))

print(x/y)
print(torch.div(x,y))
print(x.div(y))

print(torch.matmul(x,y))
print(torch.mm(x,y))

