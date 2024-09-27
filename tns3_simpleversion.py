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
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# Dropout көмегімен MLP архитектурасының анықтамасы

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # кіріс қабаты
        self.fc2 = nn.Linear(256, 128)  # жасырын қабат
        self.fc3 = nn.Linear(128, 10)  # шығыс қабаты
        self.dropout = nn.Dropout(p=0.5)  # Dropout қабаты

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # суретті векторға түрлендіру
        x = self.fc1(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.dropout(x)  # Dropout қолдану
        x = self.fc2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.dropout(x)  # Dropout қолдану
        x = self.fc3(x)
        return x


#  L1-регуляризациясы бар жоғалту функциясы


def l1_regularization(Model, Lambda_l1):
    l1_norm = sum(p.abs().sum() for p in Model.parameters())
    return Lambda_l1 * l1_norm


# Модельді, шығын функциясын және оңтайландырғышты анықтау


model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#  Модельді оқыту

epochs = 10
lambda_l1 = 0.0001  # L1-регуляризации коэффициенті
losses = []  # Шығын мәндерін сақтауға арналған тізім

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()  # градиентті нөлдеу
        outputs = model(images)  # тура тарату
        loss = criterion(outputs, labels)  # қатені есептеу
        loss += l1_regularization(model, lambda_l1)  # L1-регуляризация қосу
        loss.backward()  # кері тарату
        optimizer.step()  # салмақтарды жаңарту

        running_loss += loss.item()
    losses.append(running_loss / len(trainloader))  # Шығындардың орташа мәнін сақтаймыз
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

print(f'Accuracy: {100 * correct / total:.2f}%')


#  Қате жіктелген кескіндерді визуализациялау
def imshow(img):
    img = img / 2 + 0.5  # Де-нормализация
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


# алғашқы қате жіктелген 5 суретті классифицикациялау
for i in range(5):
    imshow(incorrect_images[i])
    print(f'Predicted: {incorrect_labels[i]}, Actual: {correct_labels[i]}')

# Шығындар кестесі
plt.plot(losses)
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
