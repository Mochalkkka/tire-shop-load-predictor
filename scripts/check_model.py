# scripts/check_model.py
import torch

checkpoint = torch.load('model.pth', map_location='cpu', weights_only=False)

print("="*60)
print("📦 ЧТО ВНУТРИ model.pth")
print("="*60)

print(f"\n Ключи в файле:")
for key in checkpoint.keys():
    print(f"   • {key}")

print(f"\n Сезонные коэффициенты:")
if 'seasonal_factors' in checkpoint:
    factors = checkpoint['seasonal_factors']
    month_names = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']
    for m in range(1, 13):
        factor = factors.get(m, 1.0)
        status = "🔺" if factor > 1.2 else "🔻" if factor < 0.8 else "◼"
        print(f"   {month_names[m-1]}: {factor:.2f} {status}")
else:
    print("   ❌ seasonal_factors ОТСУТСТВУЕТ!")

print("="*60)