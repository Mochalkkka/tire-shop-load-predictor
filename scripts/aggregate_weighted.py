# scripts/aggregate_weighted.py
import pandas as pd
import os

# 🔧 Веса запросов (сумма = 1.0)
WEIGHTS = {
    'query_tire_service': 0.35,        # шиномонтаж
    'query_tire_replacement': 0.25,    # замена шин
    'query_seasonal_change': 0.15,     # замена резины
    'query_seasonal_prep': 0.10,       # сезонная замена
    'query_wheel_balancing': 0.07,     # балансировка
    'query_tire_repair': 0.05,         # ремонт шин
    'query_wheel_maintenance': 0.03,   # переобувка
}

# Словарь месяцев
MONTHS = {
    'январь':'01','февраль':'02','март':'03','апрель':'04',
    'май':'05','июнь':'06','июль':'07','август':'08',
    'сентябрь':'09','октябрь':'10','ноябрь':'11','декабрь':'12'
}

def parse_date(period_str):
    """Превращает 'март 2024' → '2024-03-01'"""
    parts = period_str.strip().split()
    if len(parts) < 2:
        return None
    month = MONTHS.get(parts[0].lower())
    if not month:
        return None
    return f"{parts[1]}-{month}-01"

def load_query(filepath, weight):
    """Загружает файл и сразу применяет вес"""
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    df = df[['Период', 'Число запросов']]
    # Чистка чисел
    df['Число запросов'] = df['Число запросов'].astype(str).str.replace(' ', '').astype(int)
    # Применяем вес
    df['Число запросов'] = df['Число запросов'] * weight
    # Парсинг даты
    df['date'] = df['Период'].apply(parse_date)
    df = df.dropna(subset=['date'])
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'Число запросов']]

def aggregate_weighted():
    """Взвешенная агрегация всех запросов"""
    result = None
    data_raw_path = 'data_raw'
    
    for query_name, weight in WEIGHTS.items():
        filepath = os.path.join(data_raw_path, f'{query_name}.csv')
        if not os.path.exists(filepath):
            print(f"⚠️ Не найден: {filepath}")
            continue
        
        df = load_query(filepath, weight)
        df.columns = ['date', query_name]
        
        if result is None:
            result = df
        else:
            result = result.merge(df, on='date', how='outer')
        
        print(f"✅ {query_name} (вес {weight}): {len(df)} записей")
    
    # Суммируем взвешенные значения
    result = result.fillna(0)
    result['weighted_demand'] = result[list(WEIGHTS.keys())].sum(axis=1)
    result = result.sort_values('date').reset_index(drop=True)
    
    # Сохраняем в формате для обучения
    output = result[['date', 'weighted_demand']].copy()
    
    # Формируем столбец "Период" для совместимости с data_loader
    month_names_ru = ['январь','февраль','март','апрель','май','июнь',
                      'июль','август','сентябрь','октябрь','ноябрь','декабрь']
    output['Период'] = output['date'].apply(
        lambda d: f"{month_names_ru[d.month-1]} {d.year}"
    )
    
    output = output[['Период', 'weighted_demand']]
    output.columns = ['Период', 'Число запросов']
    
    # Сохраняем
    output_path = 'app/data/wordstat_dynamic.csv'
    output.to_csv(output_path, sep=';', index=False, encoding='utf-8')
    
    # Статистика для диплома
    total = result['weighted_demand'].sum()
    avg = result['weighted_demand'].mean()
    max_val = result['weighted_demand'].max()
    min_val = result['weighted_demand'].min()
    
    print(f"\n🎯 Взвешенная агрегация завершена!")
    print(f"📊 Статистика интегрального спроса:")
    print(f"   • Сумма за период: {total:.0f}")
    print(f"   • Среднее в месяц: {avg:.0f}")
    print(f"   • Диапазон: {min_val:.0f} – {max_val:.0f}")
    print(f"📁 Сохранено в: {output_path}")
    
    return output

if __name__ == "__main__":
    aggregate_weighted()