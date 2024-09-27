# Настройте MLP с использованием функции активации Leaky ReLU.
# Обучите модель на MNIST и проанализируйте результаты классификации для тестовых данных.
# Продемонстрируйте визуализацию на экране.


#  Кітапханаларды импорттау

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


#  Деректерді жүктеу және өңдеу

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


#  MLP архитектурасының анықтамасы

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # кіріс қабаты
        self.fc2 = nn.Linear(256, 128)  # жасырын қабаты
        self.fc3 = nn.Linear(128, 10)  # шығыс қабаты
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # кескінді бір өлшемді векторға кеңейту
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


#  Шығындар функциясын және оңтайландыру алгоритмін анықтау

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#  Модельді оқыту

epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()  # градиентті нөлдеу
        outputs = model(images)  # тікелей тарату
        loss = criterion(outputs, labels)  # қатені есептеу
        loss.backward()  # кері тарату
        optimizer.step()  # салмақтарды жаңарту

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

#  Модельді дәлдігін бағалау

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')


#  Қате жіктелген кескіндерді визуализациялау

def imshow(img):
    img = img / 2 + 0.5  # денормализация
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


incorrect_images = []
incorrect_labels = []
correct_labels = []

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                incorrect_images.append(images[i])
                incorrect_labels.append(predicted[i])
                correct_labels.append(labels[i])

# қате жіктелген алғашқы 5 суретті визуализиялау
for i in range(5):
    imshow(incorrect_images[i])
    print(f'Predicted: {incorrect_labels[i]}, Actual: {correct_labels[i]}')
