import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets


mnist_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean =(0.5,), std = (1.0,))])

trainset = datasets.MNIST(root='/content/',
                        train=True, download=True,
                        transform=mnist_transform)

testset = datasets.MNIST(root='/content/',
                        train=False, download=True,
                        transform=mnist_transform)

train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)

dataiter = iter(train_loader)
images, labels = next(dataiter)
print(images.shape, labels.shape)

torch_image = torch.squeeze(images[0])
print(torch_image.shape)

#여기까지가 데이터 준비

#신경망 구성
    # 레이어: 신경망의 핵심 데이터 구조로 하나 이상의 텐서를 입력받아 하나 이상의 텐서를 출력
    # 모듈: 한 개 이상의 계층이 모여서 구성
    # 모델: 한 개 이상의 모듈이 모여서 구성

import torch.nn as nn


#nn.Linear 계층 예제
input = torch.randn(128,20)

m = nn.Linear(20,30)

output = m(input)
print(output.size())

#nn.Conv2d 계층 예시
input = torch.randn(20,16,50,100)

m = nn.Conv2d(16,33,3,stride=2)
m = nn.Conv2d(16,33,(3,5),stride=(2,1), padding=(4,2))
m = nn.Conv2d(16,33,(3,5),stride=(2,1), padding=(4,2), dilation=(3,1))
print(m)

output = m(input)

print(output.size)

#Convolution Layers
#nn.Conv2d 예제
    #in_channels : channel의 개수
    #out_channels : 출력 채널의 개수
    #kernel_size : 커널(필터)의 사이즈

nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
layer = nn.Conv2d(1,20,5,1) #위에랑 같은 식
layer = nn.Conv2d(1,20,5,1).to(torch.device('cpu'))

weight = layer.weight #[20,1,5,5]

weight = weight.detach()
weight = weight.numpy()
print(weight.shape)

print(images.shape)
print(images[0].size())

input_image = torch.squeeze(images[0])
print(input_image.size)

input_data = torch.unsqueeze(images[0], dim = 0)
print(input_data.size())

output_data = layer(input_data)
output = output_data.data
output_arr = output.numpy()
print(output_arr.shape)

#Pooling layers
import torch.nn.functional as F

pool = F.max_pool2d(output, 2,2) #2개를 기준으로 Max 값을 저장해줘~ 라는 뜻
print(pool.shape)

pool_arr = pool.numpy()
print(pool_arr.shape)


#Linear Layers
    #1D 만 가능하므로 .view()를 통해 1D로 펼쳐줘야함

flatten = input_image.view(1, 28*28)
print(flatten.shape)

lin = nn.Linear(784,10)(flatten)
print(lin)

#비선형 활성화 (Non-Linear Activations)
    #softmax, ReLU, sigmoid, tanh 와 같은 활성화 함수

with torch.no_grad(): #왜 이때 no_grad가 쓰이는거임? 모델을 평가하는 단계인가?
    flatten = input_image.view(1, 28*28)
    lin = nn.Linear(784, 10)(flatten)
    softmax = F.softmax(lin, dim=1)


print(softmax) #10개의 합이 100%가 되도록 비율로 바꿔줌 (For Classification)

#ReLU함수를 적용하는 레이어
inputs = torch.randn(4,3,28,28).to(device='cpu')
print(inputs.shape)

layer = nn.Conv2d(3,20,5,1).to(device='cpu')
output = F.relu(layer(inputs))
print(output.shape)

#layer들로 구성된 신경망을 만들기 위해 다음과 같은 과정을 공부함...