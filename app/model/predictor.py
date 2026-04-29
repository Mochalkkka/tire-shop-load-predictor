# app/model/predictor.py
import torch
import numpy as np
from app.model.architecture import TimeSeriesNet

class ModelPredictor:
    def __init__(self, model_path: str):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.seq_len = checkpoint['seq_len']
        self.scaler = checkpoint['scaler']
        self.last_sequence = np.array(checkpoint['last_sequence'])
        
        self.model = TimeSeriesNet(input_size=self.seq_len)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def forecast(self, steps: int = 12) -> list:
        """Прогноз на steps месяцев вперёд"""
        predictions = []
        current_seq = self.last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(steps):
                x_input = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
                pred_norm = self.model(x_input).item()
                pred_real = self.scaler.inverse_transform([[pred_norm]])[0][0]
                predictions.append(float(pred_real))
                # Сдвиг окна
                current_seq = np.append(current_seq[1:], [pred_norm])
        
        return predictions