# app/model/ensemble.py
import numpy as np
import torch
from app.model.architecture import TimeSeriesNet
from app.model.linear_model import LinearTrendModel

class EnsemblePredictor:
    """Ансамбль из 3 моделей: PyTorch + Linear + Moving Average"""
    
    def __init__(self, model_path, linear_path):
        # 1. Загрузка PyTorch модели
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.seq_len = checkpoint['seq_len']
        self.scaler = checkpoint['scaler']
        self.last_sequence = np.array(checkpoint['last_sequence'])
        
        self.nn_model = TimeSeriesNet(input_size=self.seq_len)
        self.nn_model.load_state_dict(checkpoint['model_state_dict'])
        self.nn_model.eval()
        
        # 2. Загрузка линейной модели
        self.lr_model = LinearTrendModel.load(linear_path)
        
        # 3. История для скользящего среднего
        self.history = list(self.scaler.inverse_transform(
            self.last_sequence.reshape(-1, 1)
        ).flatten())
    
    def predict_nn(self, sequence):
        """Прогноз нейросети"""
        with torch.no_grad():
            x_input = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            pred_norm = self.nn_model(x_input).item()
            return self.scaler.inverse_transform([[pred_norm]])[0][0]
    
    def predict_lr(self, steps):
        """Прогноз линейной регрессии"""
        return self.lr_model.predict(steps=steps)
    
    def forecast(self, steps=12, weights=None):
        """
        Прогноз ансамбля
        
        weights: [nn_weight, lr_weight, ma_weight]
        """
        if weights is None:
            weights = [0.6, 0.3, 0.1]  # нейросеть важнее
        
        nn_preds = []
        lr_preds = self.predict_lr(steps)
        current_seq = self.last_sequence.copy()
        
        for i in range(steps):
            # Нейросеть
            nn_pred = self.predict_nn(current_seq)
            nn_preds.append(nn_pred)
            
            # Обновляем историю для скользящего среднего
            self.history.append(nn_pred)
        
        # Взвешенное среднее из 3 моделей
        final_preds = []
        for i in range(steps):
            nn = nn_preds[i]
            lr = lr_preds[i]
            ma = np.mean(self.history[-3:]) if len(self.history) >= 3 else nn
            
            ensemble = weights[0]*nn + weights[1]*lr + weights[2]*ma
            final_preds.append(ensemble)
        
        return final_preds