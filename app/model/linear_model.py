# app/model/linear_model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

class LinearTrendModel:
    """Линейная регрессия для прогноза тренда"""
    
    def __init__(self):
        self.model = None
        self.last_index = 0
    
    def fit(self, values):
        """Обучение на исторических данных"""
        X = np.arange(len(values)).reshape(-1, 1)
        self.model = LinearRegression()
        self.model.fit(X, values)
        self.last_index = len(values)
    
    def predict(self, steps=12):
        """Прогноз на steps месяцев вперёд"""
        if self.model is None:
            raise ValueError("Сначала вызовите fit()")
        
        future_X = np.arange(self.last_index, self.last_index + steps).reshape(-1, 1)
        return self.model.predict(future_X)
    
    def save(self, filepath):
        """Сохранить модель"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'last_index': self.last_index
            }, f)
    
    @staticmethod
    def load(filepath):
        """Загрузить модель"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        lm = LinearTrendModel()
        lm.model = data['model']
        lm.last_index = data['last_index']
        return lm