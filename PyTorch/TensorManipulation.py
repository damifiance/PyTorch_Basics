#Indexing & Slicing
import torch

x = torch.Tensor([[1,2],
                  [3,4]])

print(x)
print(x[0,0])
print(x[0,1])
print(x[1,0])
print(x[1,1])

print(x[:,0])
print(x[:,1])
print(x[0,:])
print(x[1,:])

print(x[:])

#view --> 텐서의 크기나 모양을 변경 
#기본적으로 변경 전과 후에 텐서 안의 원소 개수가 유지되어야함
#-1 로 설정되면 계산을 통해 해당 크기값을 유추

x = torch.randn(4,5)
print(x)
y = x.view(20)
print(y)
z=x.view(5,-1)
print(z)

#item --> 텐서에 값이 단 하나 존재하면 숫자값을 얻을 수 있음

x = torch.randn(1) #randn(2) 면 오류 뜸( 하나가 아니니까 )
print(x.item()) #스칼라값으로 출력됨
print(x)

x = torch.rand(3,1,3)
print(x)
print(x.shape)
print(x.squeeze())
print(x.squeeze().shape) #squeeze()는 차원 1 짜리가 있을 때만 효력을 발휘함

x = torch.rand(3,3)
print(x)
print(x.shape)

t = x.unsqueeze(dim=0)
print(t)
print(t.shape)
t = x.unsqueeze(dim=1)
print(t)
print(t.shape)
t = x.unsqueeze(dim=2)
print(t)
print(t.shape)

#stack (텐서간 결합)
x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])
print(torch.stack([x,y,z]))

# cat (concatenate) (텐서를 결합)
a = torch.randn(2,3,3)
b = torch.randn(2,3,3)

c = torch.cat((a,b), dim = 0)
print(a)
print(b)
print(c)
print(c.size())

x = torch.rand(3,6)
t1, t2, t3 = torch.chunk(x, 3, dim = 1) #몇개로 나눌거냐
print(t1, t2, t3)

x = torch.rand(3,6)
t1, t2 = torch.split(x, 3, dim = 1) #텐서의 크기가 몇인 텐서로 나눌 것이냐
