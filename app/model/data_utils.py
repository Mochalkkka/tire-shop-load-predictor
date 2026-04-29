# app/model/data_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_csv(filepath: str, seq_len: int = 12):
    """Загружает CSV, нормализует и создаёт последовательности"""
    
    # Чтение
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    df = df[['Период', 'Число запросов']]
    
    # 🔧 ИСПРАВЛЕНО: теперь поддерживаем float (для взвешенных данных)
    df['Число запросов'] = df['Число запросов'].astype(str).str.replace(' ', '')
    df['Число запросов'] = pd.to_numeric(df['Число запросов'], errors='coerce').fillna(0)
    
    # Парсинг даты
    months = {'январь':'01','февраль':'02','март':'03','апрель':'04',
              'май':'05','июнь':'06','июль':'07','август':'08',
              'сентябрь':'09','октябрь':'10','ноябрь':'11','декабрь':'12'}
    
    dates = []
    for v in df['Период']:
        p = str(v).split()
        if len(p) >= 2:
            dates.append(f"{p[1]}-{months.get(p[0].lower(), '01')}-01")
        else:
            dates.append(None)
    
    df['date'] = pd.to_datetime(dates, errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Нормализация
    values = df['Число запросов'].values.astype('float32').reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    
    # Последовательности для обучения
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
    
    return {
        'X': np.array(X),
        'y': np.array(y),
        'scaler': scaler,
        'last_sequence': scaled[-seq_len:].tolist(),
        'dates': df['date'].tolist(),
        'raw_values': df['Число запросов'].values
    }