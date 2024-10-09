# 1. Қажетті кітапханаларды импорттау
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 2. Деректерді жүктеу және зерттеу
data = pd.read_csv('Housing.csv')
print("Деректер көлемі:", data.shape)
print("\nДатасет Бағандары:", data.columns)
print("\nДеректер типі:", data.dtypes)
print("\nДеректер мәліметі:", data.info)

# 3. Сипаттар мен мақсатты айнымалыны анықтау
X = data.drop('price', axis=1)
y = data['price']

# 4. Категориялық айнымалыларды сандық мәндерег айналдыру
X = pd.get_dummies(X, drop_first=True)

# 5. Деректерді оқыту және тестілеу жиынтықтарына бөлу
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Ridge-регрессия моделін құру және оқыту
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 7. Тестілеу деректерінде болжам жасау
y_pred = model.predict(X_test)

# 8. RMSE есептеу
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f' Пәтер бағаларын болжау деректері үшін RMSE: {rmse}')

# 9. Қалдықтарды есептеу
residuals = y_test - y_pred

# 10. Қалдықтардың таралуын визуализациялау
plt.figure(figsize=(18, 5))

# 1. Қалдықтардың гистограммасы
plt.subplot(1, 3, 1)
sns.histplot(residuals, bins=30, color='blue', edgecolor='black')
plt.title('Қалдықтардың гистограммасы')
plt.xlabel('Қалдықтар')
plt.ylabel('Жиілік')

# 2. Ядерлік тығыздығы
plt.subplot(1, 3, 2)
sns.kdeplot(residuals, color='orange')
plt.title('Қалдықтардың ядерлік тығыздығы')
plt.xlabel('Қалдықтар')
plt.ylabel('Тығыздық')

# 3. Қалдықтардың KDE гистограммасы
plt.subplot(1, 3, 3)
sns.histplot(residuals, bins=30, kde=True, color='green', edgecolor='black')
plt.title('Қалдықтардың KDE гистограммасы')
plt.xlabel('Қалдықтар')
plt.ylabel('Жиілік')
plt.axvline(x=0, color='red', linestyle='--')  # Нөл нүктесінде вертикаль сызық

plt.tight_layout()
plt.show()
