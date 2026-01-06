import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim


#데이터 생성

X = torch.randn(200,1) * 10
y = X + 3 * torch.randn(200, 1)

# plt.scatter(X.numpy(), y.numpy())
# plt.ylabel('y')
# plt.xlabel('X')
# plt.grid()
# plt.show()


#모델 정의 및 파라미터

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        pred = self.linear(x)
        return pred
    

model = LinearRegressionModel()
print(model)
print(list(model.parameters()))
    # [Parameter containing:
    # tensor([[0.0440]], requires_grad=True), Parameter containing:
    # tensor([-0.9203], requires_grad=True)]

w, b = model.parameters()

w1, b1 = w[0][0].item(), b[0].item() #초기의 weight랑 bias 값.
x1 = np.array([-30,30])
y1 = w1 * x1 + b1

plt.plot(x1, y1, 'r')
plt.scatter(X,y)
plt.grid()
plt.show()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr= 0.001)

epochs = 100
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    loss.backward()

    optimizer.step()


plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

print(list(model.parameters()))

w2, b2 = model.parameters()
w_n, b_n = w2[0][0].item(), b2[0].item() #여기서 item()을 꼭 달아줘야함!!


x_n = np.array([-30, 30])
y_n = w_n * x_n + b_n

plt.plot(x_n, y_n)
plt.scatter(X,y)
plt.grid()
plt.show()
