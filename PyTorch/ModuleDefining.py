import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn


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

#-----------------------------------------------------------------------

#nn.Module 상속 클래스 정의
    #nn.Module을 상속받는 클래스 정의ㅣ
    #__init__(): 모델에서 사용될 모듈과 활성화 함수 등을 정의 // 구조를 여기서 정의하고
    #forward(): 모댈에서 실행되어야 하는 연산을 정의 // 연산을 정의

class Model(nn.Module):
    def __init__(self, inputs):
        super(Model, self).__init__()
        self.layer = nn.Linear(inputs, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)

        return x

model = Model(1)
print(list(model.children()))
    ##[Linear(in_features=1, out_features=1, bias=True), Sigmoid()]

print(list(model.modules()))
    ## 구조를 볼 수 있음 (Linear Layer, Sigmoid Activation이 있구나~~)

#-----------------------------------------------------------------------

#nn.Sequential 을 이용한 신경망 정의
    # nn.Sequential 객체로 그 안에 각 모듈을 순차적으로 실행
    # __init__()에서 사용할 네트워크 모델들을 nn.Sequential로 정의 가능
    # forward()에서 실행되어야할 계산을 가독성 높게 작성 가능


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 30, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features = 30*5*5, out_features = 10, bias = True),
            nn.ReLU(inplace = True)
        )

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer3(x)
            x = x.view(x.shape[0], -1) #텐서의 모양을 30*5*5로 잘 펴주는 것!
            x = self.layer3(x)
            return x
        

model2 = Model2()
print(list(model2.children()))
print(list(model2.modules()))

#파이토치 사전 학습 모델
#https://pytorch.org/vision/stable/models.html
