import torch

#Autograd : 자동 미분

a = torch.randn(3,3)
a = a*3
print(a)
print(a.requires_grad)

a.requires_grad_(True) #in-place 하여 변경

b = (a*a).sum()
print(b)
print(b.grad_fn) # 미분값을 계산한 함수에 대한 정보 저장

x = torch.ones(3,3, requires_grad=True)
print(x)

y = x+5

z = y*y
out = z.mean()
print(z, out)

print(out) #여기서 출력해보면 require_grad = True 라서
#out.backward()가 가능해짐!

x = torch.randn(3, requires_grad=True)

y = x*2
while y.data.norm() < 1000 :
    y = y*2

print(y)

v = torch.tensor([0.1,1.0,0.0001], dtype=torch.float)
y.backward(v) #dy/dx || x_i 자리에 v_i(v vector의 성분들) 대입

print(x.grad)

#with torch.no_grad


print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)


#detach

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

#자동 미분 흐름 예제

#계산 흐름 a -> b -> c -> out 통해 round(out)/round(a) 계산
#근데 backward()를 통해 a <- b <- c <- out 을 계산하면 round(out)/round(a) 가 a.grad에 채워짐

a = torch.ones(2,2, requires_grad=True)

print(a.data)
print(a.grad)
print(a.grad_fn)

b = a+2

c = b**2

out = c.sum()

print(out)
out.backward

print(a.data)
print(a.grad)
print(a.grad_fn) #None

print(b.data)
print(b.grad) #None
print(b.grad_fn)

print(c.data)
print(c.grad) #None
print(c.grad_fn) 

print(out.data)
print(out.grad) #None
print(out.grad_fn)

