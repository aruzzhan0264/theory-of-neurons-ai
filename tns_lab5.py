import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Жасанды деректер жиынын генерациялау
date_range = pd.date_range(start="2020-01-01", periods=1000, freq='h')
consumption_data = 100 + 10 * np.sin(np.linspace(0, 50, 1000)) + 5 * np.random.randn(1000)

# датафреймды құру және csv-да сақтау
data = pd.DataFrame({'datetime': date_range, 'consumption': consumption_data})
data.to_csv('electricity_consumption.csv', index=False)

# 2. деректерді жүктеу
data = pd.read_csv('electricity_consumption.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Деректерді уақыт қатарының форматына түрлендіру
data = data['consumption'].values

# 3. деректерді нормализациялау
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# оқыту және тестілеу деректеріне бөлу
train_size = int(len(data_normalized) * 0.8)
train_data, test_data = data_normalized[:train_size], data_normalized[train_size:]


# 4. Деректер тізбегін құру функциясы
def create_sequences(data, seq_length=24):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)


# Оқу және сынақ тізбектерін құру
X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)


# 5. GRU модель анықтамасы
class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])  # Gru соңғы шығысын қолданыңыз
        return out


# Модель параметрлары
input_size = 1  # Бір мән (тұтыну)
hidden_size = 64
output_size = 1  # Тұтынуды болжау
batch_size = 64
epochs = 25

# 6. Деректерді PyTorch үшін қажетті форматқа келтіру
X_train_tensor = torch.Tensor(X_train)  # Артық өлшемді алып тастаймыз
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)    # Артық өлшемді алып тастаймыз
y_test_tensor = torch.Tensor(y_test)

# Модельді, шығын функциясын және оңтайландырғышты инициализациялау
model = GRUNetwork(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Потеря для регрессии
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Модельді оқыту
train_losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Шығындарды визуализациялау
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 8. Модельді бағалау
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# 9. Регрессия көрсеткіштерін қолдана отырып бағалау
y_test_actual = y_test_tensor.numpy()
mse = mean_squared_error(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# 10. Болжау нәтижелерін визуализациялау
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='Actual Consumption')
plt.plot(y_pred, label='Predicted Consumption')
plt.legend()
plt.title('Electricity Consumption Prediction')
plt.xlabel('Time')
plt.ylabel('Consumption (normalized)')
plt.show()
