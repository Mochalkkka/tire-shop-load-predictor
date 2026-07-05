# scripts/calculate_accuracy.py
import sys
import os
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model.data_utils import load_and_prepare_csv
from app.model.predictor import ModelPredictor

print("="*70)
print("📊 РАСЧЁТ МЕТРИК ТОЧНОСТИ ТЕКУЩЕЙ МОДЕЛИ")
print("="*70)

# 1. Загрузка данных
print("\n📂 Загрузка данных...")
data = load_and_prepare_csv('app/data/wordstat_dynamic.csv', seq_len=12)

# Берём последние 12 месяцев как тестовую выборку
# Это реальные данные, которые модель НЕ видела при обучении
X_test_real = data['raw_values'][-12:]  # Фактические значения
dates = data['dates'][-12:]  # Даты для этих месяцев

print(f"   Тестовая выборка: последние 12 месяцев")
print(f"   Период: {dates[0].strftime('%B %Y')} — {dates[-1].strftime('%B %Y')}")
print(f"   Диапазон фактических значений: {X_test_real.min():.0f} – {X_test_real.max():.0f} запросов")

# 2. Загрузка модели
print("\n🧠 Загрузка модели...")
predictor = ModelPredictor('model.pth')
print("   ✅ Модель загружена")

# 3. Прогноз на 12 месяцев вперёд
print("\n🔮 Генерация прогноза...")
# Берём месяц начала теста
start_month = dates[0].month - 1  # 0=январь, 11=декабрь
forecast = predictor.forecast(steps=12, start_month=start_month, apply_seasonality=True)
forecast_array = np.array(forecast)

print(f"   Диапазон прогноза: {forecast_array.min():.0f} – {forecast_array.max():.0f} запросов")

# 4. Расчёт метрик
print("\n📐 Расчёт метрик...")
mae = mean_absolute_error(X_test_real, forecast_array)
rmse = np.sqrt(mean_squared_error(X_test_real, forecast_array))
mape = mean_absolute_percentage_error(X_test_real, forecast_array) * 100
accuracy = 100 - mape

# 5. Вывод результатов
print("\n" + "="*70)
print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ ТОЧНОСТИ")
print("="*70)
print(f"{'Метрика':<35} | {'Значение':<15} | {'Интерпретация'}")
print("="*70)
print(f"{'MAE (средняя абсолютная ошибка)':<35} | {mae:<15.2f} | В среднем ошибка {mae:.0f} запросов/мес")
print(f"{'RMSE (среднеквадратичная ошибка)':<35} | {rmse:<15.2f} | Учитывает крупные ошибки")
print(f"{'MAPE (средняя ошибка в %)':<35} | {mape:<15.2f}% | Относительная ошибка")
print(f"{'ТОЧНОСТЬ ПРОГНОЗА':<35} | {accuracy:<15.2f}% | 100% - MAPE")
print("="*70)

# 6. Интерпретация
print("\n📝 ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
if mape < 10:
    print("   ✅ ВЫСОКАЯ ТОЧНОСТЬ (MAPE < 10%)")
    print("   Модель пригодна для практического использования в бизнесе.")
elif mape < 20:
    print("   ✅ ХОРОШАЯ ТОЧНОСТЬ (MAPE 10-20%)")
    print("   Модель может использоваться для планирования с учётом погрешности.")
elif mape < 50:
    print("   ⚠️ УДОВЛЕТВОРИТЕЛЬНАЯ ТОЧНОСТЬ (MAPE 20-50%)")
    print("   Модель требует доработки или использования только для трендов.")
else:
    print("   ❌ НИЗКАЯ ТОЧНОСТЬ (MAPE > 50%)")
    print("   Модель непригодна для практического использования.")

# 7. Детализация по месяцам
print("\n" + "="*70)
print("📅 ДЕТАЛИЗАЦИЯ ПРОГНОЗА ПО МЕСЯЦАМ")
print("="*70)
print(f"{'Месяц':<15} | {'Факт':<10} | {'Прогноз':<10} | {'Ошибка':<10} | {'Ошибка %'}")
print("="*70)

errors_percent = []
for i in range(12):
    actual = X_test_real[i]
    predicted = forecast_array[i]
    error_abs = abs(actual - predicted)
    error_percent = (error_abs / actual * 100) if actual > 0 else 0
    errors_percent.append(error_percent)
    
    month_name = dates[i].strftime('%b %Y')
    print(f"{month_name:<15} | {actual:<10.0f} | {predicted:<10.0f} | {error_abs:<10.0f} | {error_percent:>6.2f}%")

print("="*70)
print(f"{'Средняя ошибка по модулю':<15} | {'':<10} | {'':<10} | {mae:<10.0f} | {mape:>6.2f}%")
print("="*70)

# 8. Сохранение в файл
output_file = 'model_accuracy_report.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ОТЧЁТ О ТОЧНОСТИ МОДЕЛИ ПРОГНОЗИРОВАНИЯ\n")
    f.write("="*70 + "\n\n")
    f.write(f"Период тестирования: {dates[0].strftime('%B %Y')} — {dates[-1].strftime('%B %Y')}\n")
    f.write(f"Количество месяцев: 12\n\n")
    f.write("ИТОГОВЫЕ МЕТРИКИ:\n")
    f.write(f"  MAE:  {mae:.2f} запросов\n")
    f.write(f"  RMSE: {rmse:.2f} запросов\n")
    f.write(f"  MAPE: {mape:.2f}%\n")
    f.write(f"  Точность: {accuracy:.2f}%\n\n")
    f.write("ДЕТАЛИЗАЦИЯ ПО МЕСЯЦАМ:\n")
    f.write(f"{'Месяц':<15} | {'Факт':<10} | {'Прогноз':<10} | {'Ошибка %'}\n")
    f.write("-"*70 + "\n")
    for i in range(12):
        month_name = dates[i].strftime('%b %Y')
        actual = X_test_real[i]
        predicted = forecast_array[i]
        error_percent = errors_percent[i]
        f.write(f"{month_name:<15} | {actual:<10.0f} | {predicted:<10.0f} | {error_percent:>6.2f}%\n")

print(f"\n✅ Отчёт сохранён в файл: {output_file}")
print("="*70)