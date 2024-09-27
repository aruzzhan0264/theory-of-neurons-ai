# 1 ornatu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 2  studentter jaily data kuru jane ony data frame-ge engizu
data = {
    'Имя студента': ['Aru', 'Ernar', 'Sultan', 'Uldanay', 'Aizhan', 'Nuray', 'Kanat', 'Kairat'],
    'Посещаемость занятий': [14, 11, 9, 15, 13, 10, 12, 11],
    'Всего занятий': [15, 15, 15, 15, 15, 15, 15, 15]
}
df = pd.DataFrame(data)

# 3 - katysu paiyzyn esepteu
df['Процент посещаемости'] = np.round((df['Посещаемость занятий'] / df['Всего занятий']) * 100, 2)

# 4 - katysu paiyzyna saikes oku ulgerimin korsetu
conditions = [
    (df['Процент посещаемости'] >= 90),
    (df['Процент посещаемости'] >= 75) & (df['Процент посещаемости'] < 90),
    (df['Процент посещаемости'] < 75)
]

choices = ['Өте жақсы', 'Жақсы', 'Қанағаттанарлық']
df['Успеваемость'] = np.select(conditions, choices, default='Қанағаттанарлық емес')

# 5 - student katyspagan sbasktar sanyn esepteu jane orta katysu manin tabu

df['Пропуски'] = df['Всего занятий'] - df['Посещаемость занятий']
average_attendance = np.round(df['Процент посещаемости'].mean(), 2)
df['Средняя посещаемость'] = average_attendance

# 6 - bagandardy  tolyk jane durys tartipte korsetu
df = df[['Имя студента', 'Посещаемость занятий', 'Всего занятий', 'Пропуски', 'Процент посещаемости', 'Успеваемость',
         'Средняя посещаемость']]
print(df)

# 7 - grafik quru
df.plot(kind='bar', x='Имя студента', y='Процент посещаемости', color='blue')
plt.title('Процент посещаемости студентов')
plt.xlabel('Студенты')
plt.ylabel('Процент посещаемости')
plt.show()

# 8 - группировка
students_per_performance = df.groupby('Успеваемость').size()
total_absences_by_performance = df.groupby('Успеваемость')['Посещаемость занятий'].sum()

print("\nГруппировка по успеваемости:")
print(students_per_performance)
print("\nОбщее количество посещаемости по успеваемости:")
print(total_absences_by_performance)
