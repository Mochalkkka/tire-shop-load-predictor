# app/model/predictor.py
import torch
import numpy as np
from app.model.architecture import TimeSeriesNet

class ModelPredictor:
    """
    Прогнозирование спроса с учётом сезонности.
    
    Загружает модель и сезонные коэффициенты из model.pth
    """
    
    def __init__(self, model_path: str):
        """
        Инициализация предсказателя.
        
        Args:
            model_path: Путь к файлу model.pth
        """
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.seq_len = checkpoint['seq_len']
        self.scaler = checkpoint['scaler']
        self.last_sequence = np.array(checkpoint['last_sequence'])
        
        # 🔹 Сезонные коэффициенты из данных
        self.seasonal_factors = checkpoint.get('seasonal_factors', {})
        self.overall_avg = checkpoint.get('overall_avg', 1.0)
        
        # Создание и загрузка модели
        self.model = TimeSeriesNet(input_size=self.seq_len)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def _get_seasonal_factor(self, month_idx: int) -> float:
        """
        Получить сезонный коэффициент для месяца.
        
        Args:
            month_idx: Индекс месяца (0=январь, 11=декабрь)
        
        Returns:
            Коэффициент сезонности
        """
        return self.seasonal_factors.get(month_idx + 1, 1.0)
    
    def forecast(self, steps: int = 12, start_month: int = 0, 
                 apply_seasonality: bool = True) -> list:
        """
        Прогноз на steps месяцев вперёд.
        
        Args:
            steps: Количество месяцев для прогноза
            start_month: Начальный месяц (0=январь, 11=декабрь)
            apply_seasonality: Применять сезонную коррекцию
        
        Returns:
            Список прогнозов запросов на каждый месяц
        """
        predictions = []
        current_seq = self.last_sequence.copy()
        
        with torch.no_grad():
            for i in range(steps):
                # Нормализованный вход
                x_input = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
                
                # Прогноз модели (в нормализованном виде)
                pred_norm = self.model(x_input).item()
                
                # Обратная нормализация → реальное число запросов
                pred_real = self.scaler.inverse_transform([[pred_norm]])[0][0]
                
                # 🔹 Применяем сезонный коэффициент
                if apply_seasonality:
                    month_idx = (start_month + i) % 12
                    factor = self._get_seasonal_factor(month_idx)
                    pred_real = pred_real * factor
                
                predictions.append(float(pred_real))
                
                # Сдвиг окна
                current_seq = np.append(current_seq[1:], [pred_norm])
        
        return predictions