# scripts/compare_models.py
import sys
import os

# 🔧 ВАЖНО: Добавляем корень проекта в путь
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Проверка (для отладки)
print(f"📁 Корень проекта: {project_root}")
print(f"📁 Путь к app: {os.path.join(project_root, 'app')}")

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from app.model.data_utils import load_and_prepare_csv
from app.model.linear_model import LinearTrendModel
from app.model.architecture import TimeSeriesNet

print("📊 Загрузка данных...")
data = load_and_prepare_csv('app/data/wordstat_dynamic.csv', seq_len=12)

# Последние 12 месяцев = тестовая выборка
X_test = torch.tensor(data['X'][-12:], dtype=torch.float32)
y_test = data['y'][-12:]
y_test_real = data['scaler'].inverse_transform(y_test.reshape(-1, 1)).flatten()

print(f"📈 Тестовая выборка: {len(y_test_real)} месяцев")
print(f"   Диапазон: {y_test_real.min():.0f} – {y_test_real.max():.0f} запросов\n")

# ─────────────────────────────────────────────────────────────
# 1. Нейросеть (PyTorch)
# ─────────────────────────────────────────────────────────────
print("🧠 Прогноз нейросети...")
nn_model = TimeSeriesNet(input_size=12)
checkpoint = torch.load('model.pth', map_location='cpu', weights_only=False)
nn_model.load_state_dict(checkpoint['model_state_dict'])
nn_model.eval()

with torch.no_grad():
    nn_pred_norm = nn_model(X_test).numpy()
nn_pred = data['scaler'].inverse_transform(nn_pred_norm.reshape(-1, 1)).flatten()

# ─────────────────────────────────────────────────────────────
# 2. Линейная регрессия
# ─────────────────────────────────────────────────────────────
print("📈 Прогноз линейной модели...")
lr_model = LinearTrendModel.load('linear_model.pkl')
lr_pred = lr_model.predict(steps=12)
lr_pred_real = data['scaler'].inverse_transform(lr_pred.reshape(-1, 1)).flatten()

# ─────────────────────────────────────────────────────────────
# 3. Скользящее среднее (простой базовый уровень)
# ─────────────────────────────────────────────────────────────
print("📊 Прогноз скользящего среднего...")
ma_pred = []
history = list(data['raw_values'][:-12])
for i in range(12):
    pred = np.mean(history[-3:]) if len(history) >= 3 else history[-1]
    ma_pred.append(pred)
    history.append(pred)
ma_pred = np.array(ma_pred)

# ─────────────────────────────────────────────────────────────
# 4. Ансамбль (взвешенное среднее)
# ─────────────────────────────────────────────────────────────
print("🔗 Ансамбль моделей...")
ensemble_pred = 0.6*nn_pred + 0.3*lr_pred_real + 0.1*ma_pred

# ─────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────
models = {
    'Нейросеть (PyTorch)': nn_pred,
    'Линейная регрессия': lr_pred_real,
    'Скользящее среднее': ma_pred,
    'Ансамбль (3 модели)': ensemble_pred,
}

print("\n" + "="*80)
print("📊 СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
print("Тестовая выборка: последние 12 месяцев")
print("="*80)
print(f"{'Модель':<30} | {'MAE':<12} | {'MAPE (%)':<12} | {'Точность':<10}")
print("="*80)

results = []
for name, pred in models.items():
    mae = mean_absolute_error(y_test_real, pred)
    mape = mean_absolute_percentage_error(y_test_real, pred) * 100
    accuracy = 100 - mape
    results.append((name, mae, mape, accuracy))
    print(f"{name:<30} | {mae:<12.0f} | {mape:<12.2f}% | {accuracy:<10.2f}%")

print("="*80)

# Лучшая модель
best = max(results, key=lambda x: x[3])
print(f"\n🏆 Лучшая модель: {best[0]}")
print(f"   Точность: {best[3]:.2f}% (ошибка MAPE: {best[2]:.2f}%)")

# ─────────────────────────────────────────────────────────────
# Сохранение для диплома
# ─────────────────────────────────────────────────────────────
with open('model_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ СПРОСА\n")
    f.write("="*60 + "\n\n")
    f.write("Метрики на тестовой выборке (12 месяцев):\n\n")
    f.write(f"{'Модель':<30} | {'MAE':<12} | {'MAPE (%)':<12}\n")
    f.write("="*60 + "\n")
    for name, mae, mape, acc in results:
        f.write(f"{name:<30} | {mae:<12.0f} | {mape:<12.2f}%\n")
    f.write("="*60 + "\n\n")
    f.write(f"🏆 Лучшая модель: {best[0]}\n")
    f.write(f"   Точность: {best[3]:.2f}%\n")

print("\n✅ Результаты сохранены в model_comparison.txt")