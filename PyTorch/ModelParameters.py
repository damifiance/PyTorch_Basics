#모델 파라미터
import torch
import torch.nn as nn

#Loss Function
    # 예측 값과 실제 값 사이의 오차 츩정
    # 학습이 진행됨녀서 해당 과정이 얼마나 잘 되고 있는지 나타내는 지표
    # 모델이 훈련되는 동안 촤소화될 값으로 주어진 문제에 대한 성공 지표
    # 손실 하함수에 따른 결과를 통해 학습 파라미터를 조정
    # 최적화 이론에서 최소화 하고자 하는 함수
    # 미분 가능한 함수 사용
    # 파이토치의 주요 손실 함수
        # torch.nn.BCELoss (BinaryCrossEntropy) : 이진 분류를 위해 사용
        # torch.nn.CrossEntropyLoss : 다중 클래스 분류를 위해 사용
        # torch.nn.MSELoss (MeanSquaredError) : 회귀 모델에서 사용

criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss() #이런식으로 호출해서 쓰면 됨

#Optimizer
    #손실 함수를 기반으로 모델이 어떻게 업데이트 되어야 하는지 결정 (특정 종류의 확률적 경사 하강법 구현)
    #optimizer는 step()을 통해 전달받은 파라미터를 모델 업데이트
    #zero_grad()를 이용해 옵티마이저에 사용된 파라미터들의 기울기를 0으로 설정
    #torch.optim.lr_scheduler를 이용해 epochs 에 따라 learning rate 설정
    # 파이토치의 주요 Optimizer
        #Adadelta, Adagrad, Adam, RMSprop, SGD #optim.(Name_of_Optimizer) 이런식으로 호출

#Learning rate Scheduler
    #학습시 특정 조건에 따라 학습률을 조정하여 최적화 진행
    #일정 횟수 이상이 되면 학습률을 감소시키거나 전역 최저점 근처에 가면 학습률을 줄이는 등..
    #파이토치의 학습률 스케줄러 종류 (자료 참고)
        #LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau (optim.lr_scheduler.(Name_of_scheduler)로 호출)

#지표(Metrics)
    #모델의 학습과 테스트 단계를 모니터링
import torchmetrics

preds = torch.randn(10,5).softmax(dim = -1)
target = torch.randint(5, (10,))
print(preds, target)

acc = torchmetrics.functional.accuracy(preds, target, task='multiclass', num_classes=5)
print(acc)

metric = torchmetrics.Accuracy(task='multitask', num_classes=5)

n_batches = 10
for i in range(n_batches):
    preds = torch.randn(10,5).sofmax(dim=-1)
    target = torch.randint(5,(10,))

    acc = metric(preds, target)
    print(acc)

acc = metric.compute()
print(acc)