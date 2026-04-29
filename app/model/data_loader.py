import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    df = df[['Период', 'Число запросов']]
    
    # Чистка чисел
    df['Число запросов'] = df['Число запросов'].astype(str).str.replace(' ', '').astype(int)
    
    # Парсинг даты
    months = {
        'январь': '01', 'февраль': '02', 'март': '03', 'апрель': '04',
        'май': '05', 'июнь': '06', 'июль': '07', 'август': '08',
        'сентябрь': '09', 'октябрь': '10', 'ноябрь': '11', 'декабрь': '12'
    }
    
    new_dates = []
    for value in df['Период']:
        parts = value.split()
        month = months[parts[0].lower()]
        year = parts[1]
        new_dates.append(f"{year}-{month}-01")
    
    df['date'] = pd.to_datetime(new_dates)
    df.set_index('date', inplace=True)
    df.rename(columns={'Число запросов': 'requests'}, inplace=True)
    
    return df[['requests']]