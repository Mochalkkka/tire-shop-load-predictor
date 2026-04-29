# test_data.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('app/data/wordstat_dynamic.csv', sep=';', encoding='utf-8')
df['Число запросов'] = df['Число запросов'].astype(str).str.replace(' ', '').astype(int)

# Простой график
plt.figure(figsize=(12, 4))
plt.plot(df['Период'], df['Число запросов'], marker='o')
plt.title('История запросов')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Статистика
print(f"Среднее: {df['Число запросов'].mean():.0f}")
print(f"Максимум: {df['Число запросов'].max():.0f} в {df.loc[df['Число запросов'].idxmax(), 'Период']}")