import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

import torch.optim as optim

#gpu 설정 (나는 안되긴 함ㅋㅋ)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean= (0.5,), std= (0.5, ))])

trainset = datasets.FashionMNIST(root= '/content/',
                                 train = True, download = True,
                                 transform=transform)


testset = datasets.FashionMNIST(root= '/content/',
                                 train = False, download = True,
                                 transform=transform)


train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

labels_map = {
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneakers',
    8: 'Bag',
    9: 'Ankle Boots'
}

# figure = plt.figure(figsize=(12,12))
# cols, rows = 4,4

# for i in range(1, cols*rows + 1):
#     image= images[i].squeeze()
#     label_idx = labels[i].item()
#     label = labels_map[label_idx]

#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis('off')
#     plt.imshow(image, cmap='gray')

# plt.show()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*5*5, 120) #할 떄마다 in_features out_features 잘 맞춰줄것!
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10) #마지막에 출력을 10개 (0~9까지 10가지)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

net = NeuralNet()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)

total_batch = len(train_loader)

for epoch in range(10):
    running_loss = 0.0

    for i , data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch+1, i+1, running_loss/2000))

#모델의 저장 및 로드
PATH = './fashion_mnist.pth'
torch.save(net.state_dict(), PATH)

net = NeuralNet()
net.load_state_dict(torch.load(PATH))


# def imshow(image):
#     image = image / 2 + 0.5
#     npimg = image.numpy()

#     fig = plt.figure(figsize=(16,8))
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()


# import torchvision

# dataiter = iter(test_loader)
# images, labels = dataiter.next()


outputs = net(images)

_, predicted = torch.max(outputs, 1)
print(predicted)

print(''.join('{}, '.format(labels_map[int(predicted[j].numpy())]) for j in range(6))) 

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(100 * correct / total)