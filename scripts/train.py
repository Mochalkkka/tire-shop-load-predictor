# scripts/train.py
import sys, os, torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Добавляем путь к app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model.architecture import TimeSeriesNet
from app.model.data_utils import load_and_prepare_csv

# Пути
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "app", "data", "wordstat_dynamic.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pth")

print("📊 Загрузка данных...")
data = load_and_prepare_csv(CSV_PATH, seq_len=12)

X = torch.tensor(data['X'], dtype=torch.float32)
y = torch.tensor(data['y'], dtype=torch.float32)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("🧠 Создание модели...")
model = TimeSeriesNet(input_size=12, hidden_size=32)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🚀 Обучение (500 эпох)...")
for epoch in range(500):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/500, Loss: {total_loss/len(loader):.4f}")

# Сохранение
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': data['scaler'],
    'seq_len': 12,
    'last_sequence': data['last_sequence']
}, MODEL_PATH)

# 🔹 Сохраняем линейную модель (добавь это в конец train.py)
from app.model.linear_model import LinearTrendModel

print("\n📈 Обучение линейной модели...")
lm = LinearTrendModel()
lm.fit(data['y'])  #  Просто передаём массив
lm.save(os.path.join(PROJECT_ROOT, 'linear_model.pkl'))
print(f"✅ Линейная модель сохранена: linear_model.pkl")

print(f"✅ Модель сохранена: {MODEL_PATH}")
print(f"📏 Размер файла: {os.path.getsize(MODEL_PATH) // 1024} КБ")