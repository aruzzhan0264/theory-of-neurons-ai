import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из CSV-файла
df = pd.read_csv('AI_csv.csv')
print(df.head())
print(df.columns)

# Очистка данных: проверка и удаление пропусков
initial_shape = df.shape
df.dropna(inplace=True)
print(f"Удалено {initial_shape[0] - df.shape[0]} строк с пропущенными значениями.")

# Удаление пробелов в названиях колонок
df.columns = df.columns.str.strip()

# Проверка наличия необходимых колонок и расчет процента посещаемости
attended_col = 'Overall Lect.'
total_col = 'Overall Lect.'

if attended_col in df.columns and total_col in df.columns:
    df['Процент посещаемости'] = np.round((df[attended_col].astype(int) / df[total_col].astype(int)) * 100, 2)

# Создание колонки с днями недели
date_columns = df.columns[2:12]
days_of_week = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб'] * 2

day_of_week_column = []
for col in date_columns:
    day_of_week_column.extend(days_of_week[:len(df)])

df['День недели'] = day_of_week_column[:len(df)]

# Классификация успеваемости
conditions = [
    (df['Процент посещаемости'] >= 90),
    (df['Процент посещаемости'] >= 75) & (df['Процент посещаемости'] < 90),
    (df['Процент посещаемости'] < 75)
]
choices = ['Отлично', 'Хорошо', 'Удовлетворительно']
df['Успеваемость'] = np.select(conditions, choices, default='Неудовлетворительно')

# Визуализация распределения посещаемости по дням недели
plt.figure(figsize=(8, 6))
sns.countplot(x='День недели', hue='Успеваемость', data=df, palette='viridis')
plt.title('Распределение посещаемости по дням недели')
plt.xticks(rotation=45)
plt.legend(title='Успеваемость')
plt.show()

# Визуализация временных рядов посещаемости студентов
plt.figure(figsize=(8, 6))
if 'Student Name' in df.columns:
    df.set_index('Student Name')['Процент посещаемости'].plot()
    plt.title('Процент посещаемости по студентам')
    plt.xlabel('Студенты')
    plt.ylabel('Процент посещаемости')
    plt.show()

# Группировка по успеваемости
students_per_performance = df.groupby('Успеваемость').size()
total_attendance_by_performance = df.groupby('Успеваемость')[attended_col].sum()

print("\nГруппировка по успеваемости:")
print(students_per_performance)
print("\nОбщее количество посещаемости по успеваемости:")
print(total_attendance_by_performance)
