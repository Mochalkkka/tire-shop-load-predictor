# scripts/train.py
import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model.architecture import TimeSeriesNet
from app.model.data_utils import load_and_prepare_csv
from app.model.linear_model import LinearTrendModel

# Пути
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "app", "data", "wordstat_dynamic.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pth")
LINEAR_PATH = os.path.join(PROJECT_ROOT, "linear_model.pkl")

print("="*60)
print("🧠 ОБУЧЕНИЕ МОДЕЛИ ПРОГНОЗИРОВАНИЯ")
print("="*60)

# 1. Загрузка данных
print("\n📊 Загрузка данных...")
data = load_and_prepare_csv(CSV_PATH, seq_len=12)

print(f"   Всего записей: {len(data['y']) + 12}")
print(f"   Примеров для обучения: {len(data['X'])}")
print(f"   Диапазон значений: {data['raw_values'].min():.0f} – {data['raw_values'].max():.0f}")

# 2. Подготовка данных для PyTorch
X = torch.tensor(data['X'], dtype=torch.float32)
y = torch.tensor(data['y'], dtype=torch.float32)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Создание модели
print("\n🧠 Создание модели...")
model = TimeSeriesNet(input_size=12, hidden_size=32)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"   Архитектура: 12 → 32 → 16 → 1")
print(f"   Функция потерь: MSE")
print(f"   Оптимизатор: Adam (lr=0.001)")

# 4. Обучение
print("\n🚀 Обучение (500 эпох)...")
for epoch in range(500):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 100 == 0:
        avg_loss = total_loss / len(loader)
        print(f"   Epoch {epoch+1}/500, Loss: {avg_loss:.4f}")

# 5. 🔹 Вычисляем сезонные коэффициенты из данных
print("\n📊 Вычисление сезонных коэффициентов...")

df_raw = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')
df_raw['Число запросов'] = pd.to_numeric(
    df_raw['Число запросов'].astype(str).str.replace(' ', ''), 
    errors='coerce'
).fillna(0)

# Словарь месяцев
months_dict = {'январь':'01','февраль':'02','март':'03','апрель':'04',
               'май':'05','июнь':'06','июль':'07','август':'08',
               'сентябрь':'09','октябрь':'10','ноябрь':'11','декабрь':'12'}

def parse_month(period_str):
    parts = str(period_str).split()
    return months_dict.get(parts[0].lower(), '01') if len(parts) >= 2 else '01'

df_raw['month'] = df_raw['Период'].apply(parse_month).astype(int)

# Среднее по месяцам
monthly_avg = df_raw.groupby('month')['Число запросов'].mean()
overall_avg = df_raw['Число запросов'].mean()

# 🔹 УСИЛЕННЫЕ сезонные коэффициенты (возводим в степень 1.5)
# Это делает пики выше, а спады глубже
raw_factors = (monthly_avg / overall_avg).to_dict()
seasonal_factors = {m: f ** 1.5 for m, f in raw_factors.items()}

# Нормализуем, чтобы среднее было 1.0
avg_factor = sum(seasonal_factors.values()) / 12
seasonal_factors = {m: f / avg_factor for m, f in seasonal_factors.items()}

print("   Сезонные коэффициенты (усиленные):")
month_names = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']
for m in range(1, 13):
    factor = seasonal_factors.get(m, 1.0)
    status = "🔺" if factor > 1.2 else "🔻" if factor < 0.8 else "◼"
    print(f"   {month_names[m-1]}: {factor:.2f} {status}")
# 6. Сохранение модели PyTorch (с сезонностью!)
print("\n💾 Сохранение модели...")
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': data['scaler'],
    'seq_len': 12,
    'last_sequence': data['last_sequence'],
    'seasonal_factors': seasonal_factors,  # ← Сезонность
    'overall_avg': overall_avg,  # ← Среднее значение
}, MODEL_PATH)

print(f"✅ Модель PyTorch сохранена: {MODEL_PATH}")
print(f"📏 Размер файла: {os.path.getsize(MODEL_PATH) // 1024} КБ")

# 7. Обучение линейной модели
print("\n📈 Обучение линейной модели...")
lm = LinearTrendModel()
lm.fit(data['y'])
lm.save(LINEAR_PATH)
print(f"✅ Линейная модель сохранена: {LINEAR_PATH}")

# 8. Итог
print("\n" + "="*60)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("="*60)
print(f"\n📁 Файлы:")
print(f"   • {MODEL_PATH}")
print(f"   • {LINEAR_PATH}")
print(f"\n🚀 Для запуска сервера:")
print(f"   uvicorn app.main:app --reload")
print("="*60)