import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import sys
import matplotlib.pyplot as plt

# Установка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Функция для подавления вывода
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Загрузка и преобразование данных CIFAR-10
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    with SuppressPrint():
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return trainloader, testloader


# Определение архитектуры свёрточной сети с переменным размером ядра
class CNN(nn.Module):
    def __init__(self, kernel_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Функция для тренировки модели
def train_model(kernel_size, trainloader, testloader, epochs=10):
    print(f'Training model with kernel size {kernel_size}x{kernel_size}')
    model = CNN(kernel_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Средняя потеря на одну эпоху
        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)

        # Оценка точности после каждой эпохи
        accuracy = evaluate_model(model, testloader)
        test_accuracies.append(accuracy)

        print(
            f'Epoch [{epoch + 1}/{epochs}], Kernel size: {kernel_size}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%'
        )

    return model, train_losses, test_accuracies


# Функция для оценки точности модели
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# Функция для визуализации потерь и точности
def plot_metrics(kernel_size, train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # График потерь
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label=f'Kernel size {kernel_size}')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label=f'Kernel size {kernel_size}')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()


# Основная программа


if __name__ == "__main__":
    # Загрузка данных без вывода лишней информации
    trainloader, testloader = load_data()

    kernel_sizes = [3, 5, 7]  # список размеров ядер, которые нужно протестировать

    # Обучение моделей с разными размерами ядер
    for kernel_size in kernel_sizes:
        model, train_losses, test_accuracies = train_model(kernel_size, trainloader, testloader)
        plot_metrics(kernel_size, train_losses, test_accuracies)
