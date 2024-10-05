import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Деректерді өңдеу
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 2. Оқу және валидация деректерін бөлу
train_data, val_data = train_test_split(trainset, test_size=0.2, random_state=42)
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
valloader = DataLoader(val_data, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# 3. MLP архитектурасы
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # кіріс қабаты
        self.fc2 = nn.Linear(256, 128)  # жасырын қабат
        self.fc3 = nn.Linear(128, 10)  # шығыс қабаты
        self.dropout = nn.Dropout(p=0.5)  # Dropout қолдану

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # суретті векторға түрлендіру
        x = nn.LeakyReLU(0.01)(self.fc1(x))
        x = self.dropout(x)
        x = nn.LeakyReLU(0.01)(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 4. L1-регуляризация функциясы
def l1_regularization(Model, Lambda_l1):
    l1_norm = sum(p.abs().sum() for p in Model.parameters())
    return Lambda_l1 * l1_norm


# 5. Модельді, шығын функциясын және оңтайландырғышты анықтау
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Модельді оқыту (MNIST)
epochs = 5
lambda_l1 = 0.001  # L1-регуляризация коэффициенті
train_losses = []
val_losses = []

for epoch in range(epochs):
    # 6.1. Оқу
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss += l1_regularization(model, lambda_l1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(trainloader))

    # 6.2. Валидация
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in valloader:
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item()

    val_losses.append(val_loss / len(valloader))
    print(f'{epoch + 1}-эпоха, Оқу жиыны шығыны: {train_losses[-1]:.4f}, Валидация жиыны шығыны: {val_losses[-1]:.4f}')

# 7. Шығындар графигін визуализациялау
plt.plot(train_losses, label='Оқу шығыны')
plt.plot(val_losses, label='Валидация шығыны')
plt.legend()
plt.show()

# 8. Модельдің дәлдігін бағалау
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Дәлдік: {100 * correct / total:.2f}%')


# 9. Қате жіктелген кескіндерді визуализациялау
def imshow(img):
    img = img / 2 + 0.5  # де-нормализация
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

for i in range(5):
    imshow(incorrect_images[i])
    print(f'Болжам: {incorrect_labels[i]}, Шынайы: {correct_labels[i]}')

# 10. XOR міндетіне арналған модельді оқыту
xor_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
xor_labels = torch.tensor([0, 1, 1, 0], dtype=torch.float32).unsqueeze(1)


class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 2 кіріс
        self.fc2 = nn.Linear(4, 1)  # 1 шығыс

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


xor_model = XORNet()
xor_criterion = nn.MSELoss()
xor_optimizer = optim.SGD(xor_model.parameters(), lr=0.1)

xor_epochs = 10000
for epoch in range(xor_epochs):
    xor_optimizer.zero_grad()
    xor_outputs = xor_model(xor_data)
    xor_loss = xor_criterion(xor_outputs, xor_labels)
    xor_loss.backward()
    xor_optimizer.step()

    if epoch % 1000 == 0:
        print(f'{epoch}-эпоха, Шығын: {xor_loss.item():.4f}')
