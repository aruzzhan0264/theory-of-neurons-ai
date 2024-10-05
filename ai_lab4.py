# Кітапханаларды импорттау
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# MNIST деректерін жүктеу және дайындау (8.1 бөлігі)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# МНIST үшін MLP архитектурасы (8.1 бөлігі)
class MnistMlp(nn.Module):
    def __init__(self):
        super(MnistMlp, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Кескіндерді векторға айналдыру
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


# XOR деректерін дайындау (8.2 бөлігі)
XOR_X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_y = torch.Tensor([[0], [1], [1], [0]])


# XOR үшін MLP архитектурасы (8.2 бөлігі)
class XorMlp(nn.Module):
    def __init__(self):
        super(XorMlp, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # XOR үшін ақырғы нәтиже сигмоидпен
        return x


# Модельдерді құру
mnist_model = MnistMlp()
xor_model = XorMlp()

# Шығын функциясы және оңтайландырғыштар
mnist_criterion = nn.CrossEntropyLoss()
xor_criterion = nn.BCELoss()  # XOR үшін Binary Cross-Entropy Loss
mnist_optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)
xor_optimizer = optim.Adam(xor_model.parameters(), lr=0.01)

# MNIST моделін оқыту (8.1 бөлігі)
mnist_epochs = 5
mnist_losses = []
for epoch in range(mnist_epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        mnist_optimizer.zero_grad()
        outputs = mnist_model(images)
        loss = mnist_criterion(outputs, labels)
        loss.backward()
        mnist_optimizer.step()
        running_loss += loss.item()
    mnist_losses.append(running_loss / len(trainloader))
    print(f'Эпоха {epoch + 1}/{mnist_epochs}, MNIST шығыны: {running_loss / len(trainloader):.4f}')

# XOR моделін оқыту (8.2 бөлігі)
xor_epochs = 100  # Epoch саны 100-ге дейін азайтылды
xor_losses = []
for epoch in range(xor_epochs):
    xor_optimizer.zero_grad()
    outputs = xor_model(XOR_X)
    loss = xor_criterion(outputs, XOR_y)
    loss.backward()
    xor_optimizer.step()
    xor_losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Эпоха {epoch + 1}/{xor_epochs}, XOR шығыны: {loss.item():.4f}')

# Графиктерді визуализациялау
plt.figure(figsize=(12, 6))

# MNIST шығындары графигі (8.1 бөлігі)
plt.subplot(1, 2, 1)
plt.plot(mnist_losses)
plt.title('MNIST Эпоха бойынша шығындар')
plt.xlabel('Эпоха')
plt.ylabel('Шығын')

# XOR шығындары графигі (8.2 бөлігі)
plt.subplot(1, 2, 2)
plt.plot(xor_losses)
plt.title('XOR Эпоха бойынша шығындар')
plt.xlabel('Эпоха')
plt.ylabel('Шығын')

plt.show()
