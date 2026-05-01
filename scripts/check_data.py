# scripts/check_data.py
import pandas as pd
import sys, os

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 АНАЛИЗ ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
print("="*60)

# Загружаем файл, на котором обучалась модель
df = pd.read_csv('app/data/wordstat_dynamic.csv', sep=';', encoding='utf-8')
df['Число запросов'] = pd.to_numeric(df['Число запросов'].astype(str).str.replace(' ', ''), errors='coerce').fillna(0)

# Парсинг дат
months = {'январь':'01','февраль':'02','март':'03','апрель':'04',
          'май':'05','июнь':'06','июль':'07','август':'08',
          'сентябрь':'09','октябрь':'10','ноябрь':'11','декабрь':'12'}

def parse_date(period_str):
    parts = str(period_str).split()
    if len(parts) >= 2:
        return f"{parts[1]}-{months.get(parts[0].lower(), '01')}-01"
    return None

df['date'] = pd.to_datetime(df['Период'].apply(parse_date), errors='coerce')
df = df.dropna(subset=['date']).sort_values('date')

# Статистика
print(f"\n📁 Файл: app/data/wordstat_dynamic.csv")
print(f"📊 Всего записей: {len(df)}")
print(f"📅 Период: {df['date'].min().strftime('%B %Y')} — {df['date'].max().strftime('%B %Y')}")
print(f"🔢 Диапазон запросов: {df['Число запросов'].min():.0f} — {df['Число запросов'].max():.0f}")
print(f"📈 Среднее: {df['Число запросов'].mean():.0f} запросов/месяц")

# Сколько лет данных
date_range = df['date'].max() - df['date'].min()
years = date_range.days / 365
print(f"⏳ Длительность: {years:.1f} лет")

# Сезонность (проверка)
df['month'] = df['date'].dt.month
monthly_avg = df.groupby('month')['Число запросов'].mean()
print(f"\n🔄 Среднее по месяцам (сезонность):")
month_names = ['янв','фев','мар','апр','май','июн','июл','авг','сен','окт','ноя','дек']
for m in range(1, 13):
    if m in monthly_avg.index:
        print(f"   {month_names[m-1]}: {monthly_avg[m]:.0f}")

# Как данные использовались в обучении
SEQ_LEN = 12  # окно модели
train_size = len(df) - SEQ_LEN - 12  # -12 для теста
print(f"\n🧠 Использование в обучении:")
print(f"   • Всего точек: {len(df)}")
print(f"   • Обучающая выборка: {train_size} примеров")
print(f"   • Тестовая выборка: 12 месяцев (последние)")
print(f"   • Окно модели (seq_len): {SEQ_LEN} месяцев")

print("\n✅ Вывод: модель обучалась на данных с " + 
      f"{df['date'].min().strftime('%B %Y')} по {df['date'].max().strftime('%B %Y')}")