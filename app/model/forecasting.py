import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def forecast_with_trend(data, start_date, end_date):
    data = data.copy()
    data['t'] = np.arange(len(data))
    
    # Тренд
    model = LinearRegression()
    model.fit(data[['t']], data['requests'])
    
    # Сезонность
    data['month'] = data.index.month
    monthly_avg = data.groupby('month')['requests'].mean()
    overall_avg = data['requests'].mean()
    seasonal_index = monthly_avg / overall_avg
    
    # Прогноз
    future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    t_future = np.arange(len(data), len(data) + len(future_dates))
    trend_future = model.predict(t_future.reshape(-1, 1))
    
    forecast_values = []
    for i, date in enumerate(future_dates):
        month = date.month
        predicted = trend_future[i] * seasonal_index.get(month, 1.0)
        forecast_values.append(predicted)
    
    return pd.DataFrame({
        'date': future_dates,
        'forecast_requests': forecast_values
    })