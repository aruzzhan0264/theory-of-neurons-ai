# кітапханаларды жүктеу

import numpy as np
import matplotlib.pyplot as plt


# Активация функциясы

def activation_function(x):
    return np.where(x >= 0, 1, 0)


# Бір қабатты перцептрон

class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=1000):
        self.W = np.zeros(input_size + 1)  # +1 bias үшін
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        x = np.insert(x, 0, 1)  # bias қосу
        return activation_function(np.dot(self.W, x))

    def fit(self, x, y):
        for _ in range(self.epochs):
            for i in range(len(x)):
                x_i = np.insert(x[i], 0, 1)  # bias қосу
                y_hat = activation_function(np.dot(self.W, x_i))
                self.W += self.lr * (y[i] - y_hat) * x_i


# классификация мәліметтерін генерациялау
def generate_data(n=100):
    X = np.linspace(-2 * np.pi, 2 * np.pi, n)
    y = np.sin(X)
    labels = np.where(y >= 0, 1, 0)  # метки для точек относительно синусоиды
    return X, labels


# Нәтижерін визуализациялау

def plot_results(X, labels, perceptron):
    plt.scatter(X, labels, c=labels, cmap='coolwarm')
    plt.plot(X, np.sin(X), label='y=sin(x)', color='green')
    predicted_labels = [perceptron.predict([x]) for x in X]
    plt.scatter(X, predicted_labels, marker='x', color='black', label='Предсказанные метки')

    plt.xlabel('Значения X')
    plt.ylabel('Метки')
    plt.legend()
    plt.show()


# Нақтылық бағасы

def calculate_accuracy(X, labels, perceptron):
    predicted_labels = [perceptron.predict([x]) for x in X]
    accuracy = np.mean(predicted_labels == labels)
    return accuracy


#  негізгі функция
def main():
    X, labels = generate_data()
    X = X.reshape(-1, 1)

    perceptron = Perceptron(input_size=1)
    perceptron.fit(X, labels)

    plot_results(X, labels, perceptron)


if __name__ == '__main__':
    main()
