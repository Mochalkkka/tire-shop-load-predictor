# scripts/check_seasonality.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('app/data/wordstat_dynamic.csv', sep=';', encoding='utf-8')
df['Число запросов'] = pd.to_numeric(df['Число запросов'].astype(str).str.replace(' ', ''), errors='coerce')

# Среднее по месяцам (сезонность)
df['месяц'] = df['Период'].apply(lambda x: x.split()[0].lower())
month_order = ['январь','февраль','март','апрель','май','июнь','июль','август','сентябрь','октябрь','ноябрь','декабрь']
monthly = df.groupby('месяц')['Число запросов'].mean().reindex(month_order)

print("📊 СРЕДНЯЯ СЕЗОННОСТЬ ПО ДАННЫМ:")
for month, val in monthly.items():
    print(f"   {month.capitalize()}: {val:.0f}")

# График
plt.figure(figsize=(12, 4))
plt.bar(month_order, monthly.values, color='#333')
plt.title('Сезонность в исходных данных (среднее по месяцам)')
plt.xticks(rotation=45)
plt.ylabel('Запросов')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('seasonality_check.png', dpi=150)
print("\n✅ График сохранён: seasonality_check.png")