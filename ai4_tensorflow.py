import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# 1. Деректерді жүктеу және алдын ала өңдеу
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 0-1 диапазонына нормализация

# 2. Модель архитектурасын құру
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Суреттерді векторға түрлендіру
    layers.Dense(128, activation='relu'),   # Жасырын қабат
    layers.Dropout(0.2),                    # Dropout қабаты
    layers.Dense(10, activation='softmax')  # Шығыс қабаты
])

# 3. Модельді компиляциялау
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Модельді оқыту
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# 5. Модельдің дәлдігін бағалау
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Дәлдік: {test_acc * 100:.2f}%')

# 6. Қате жіктелген кескіндерді визуализациялау
predictions = model.predict(x_test)
incorrect_indices = np.where(np.argmax(predictions, axis=1) != y_test)[0]

for i in range(5):
    plt.imshow(x_test[incorrect_indices[i]], cmap='gray')
    plt.title(f'Болжам: {np.argmax(predictions[incorrect_indices[i]])}, Шынайы: {y_test[incorrect_indices[i]]}')
    plt.axis('off')
    plt.show()

# 7. Тренировка барысындағы шығын мен дәлдікті визуализациялау
plt.figure(figsize=(12, 4))

# Шығын графигі
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Оқыту шығыны')
plt.plot(history.history['val_loss'], label='Валидация шығыны')
plt.title('Шығын графигі')
plt.xlabel('Эпоха')
plt.ylabel('Шығын')
plt.legend()

# Дәлдік графигі
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Оқыту дәлдігі')
plt.plot(history.history['val_accuracy'], label='Валидация дәлдігі')
plt.title('Дәлдік графигі')
plt.xlabel('Эпоха')
plt.ylabel('Дәлдік')
plt.legend()

plt.tight_layout()
plt.show()

# 8. XOR міндетіне арналған модельді оқыту
xor_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
xor_labels = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 9. XOR моделі
xor_model = keras.Sequential([
    layers.Dense(4, activation='sigmoid', input_shape=(2,)),  # 2 кіріс
    layers.Dense(1, activation='sigmoid')                     # 1 шығыс
])

# 10. XOR моделін компиляциялау
xor_model.compile(optimizer='sgd', loss='mean_squared_error')

# 11. XOR моделін оқыту
xor_history = xor_model.fit(xor_data, xor_labels, epochs=100, verbose=1)

# 12. XOR шығысын бағалау
xor_loss = xor_model.evaluate(xor_data, xor_labels, verbose=0)
print(f'XOR Шығын: {xor_loss:.4f}')

# 13. XOR тренировка барысындағы шығынды визуализациялау
plt.figure()
plt.plot(xor_history.history['loss'], label='XOR шығыны')
plt.title('XOR шығыны графигі')
plt.xlabel('Эпоха')
plt.ylabel('Шығын')
plt.legend()
plt.show()