# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import os

from app.model.predictor import ModelPredictor
from app.model.smo import find_optimal_n, mmn_metrics

app = FastAPI(title="Прогноз загрузки шиномонтажа")

# 🔹 Подключение статических файлов
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 🔹 Шаблоны
templates = Jinja2Templates(directory="app/templates")

# 🔹 Загрузка модели
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pth")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Файл модели не найден: {MODEL_PATH}\nЗапусти: python scripts/train.py")

predictor = ModelPredictor(MODEL_PATH)

# 🔹 Главная страница
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

# 🔹 Страница результатов
@app.get("/predict")
def predict_page(
    request: Request,
    boxes: int = 3,
    mu: float = 3.0,
    start_date: str = None,
    end_date: str = None,
    conversion: float = 10.0,
    work_hours: int = 10
):
    # Парсинг дат
    if not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Укажите даты начала и окончания прогноза")
    
    try:
        start_dt = datetime.strptime(start_date + "-01", "%Y-%m-%d")
        end_dt = datetime.strptime(end_date + "-01", "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Неверный формат даты. Используйте YYYY-MM")
    
    if end_dt < start_dt:
        raise HTTPException(status_code=400, detail="Дата окончания должна быть после даты начала")
    
    months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    
    if months_diff < 1:
        raise HTTPException(status_code=400, detail="Минимальный период: 1 месяц")
    if months_diff > 24:
        raise HTTPException(status_code=400, detail="Максимальный период: 24 месяца")
    
    # Прогноз
    forecast_requests = predictor.forecast(
        steps=months_diff + 1, 
        start_month=start_dt.month - 1, 
        apply_seasonality=True
    )
    
    # Расчёт СМО
    results = []
    month_names = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
                   "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
    
    current_dt = start_dt
    days_in_month = 30
    
    for requests in forecast_requests:
        clients_per_month = requests * (conversion / 100)
        lambda_hour = clients_per_month / (work_hours * days_in_month)
        n_opt, metrics = find_optimal_n(lambda_hour, mu)
        
        results.append({
            "month_name": f"{month_names[current_dt.month - 1]} {current_dt.year}",
            "forecast_requests": round(requests),
            "lambda_per_hour": round(lambda_hour, 4),
            "optimal_boxes": n_opt if n_opt else 1,
            "avg_wait_minutes": round(metrics["Wq"] * 60, 2) if metrics else 0,
            "utilization_percent": round(metrics["rho"] * 100, 1) if metrics else 0
        })
        
        # Следующий месяц
        if current_dt.month == 12:
            current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
            current_dt = current_dt.replace(month=current_dt.month + 1)
    
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "request": request,
            "results": results,
            "params": {
                "boxes": boxes,
                "mu": mu,
                "start_date": start_date,
                "end_date": end_date,
                "work_hours": work_hours,
                "conversion": conversion
            }
        }
    )

# 🔹 API эндпоинт
@app.get("/api/predict")
def predict_api(
    boxes: int = 3,
    mu: float = 3.0,
    start_date: str = None,
    end_date: str = None,
    conversion_rate: float = 0.1,
    work_hours: int = 10,
    days_in_month: int = 30
):
    if not start_date or not end_date:
        return {"error": "Укажите start_date и end_date в формате YYYY-MM"}
    
    start_dt = datetime.strptime(start_date + "-01", "%Y-%m-%d")
    end_dt = datetime.strptime(end_date + "-01", "%Y-%m-%d")
    months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    
    forecast_requests = predictor.forecast(
        steps=months_diff + 1,
        start_month=start_dt.month - 1,
        apply_seasonality=True
    )
    
    results = []
    current_dt = start_dt
    month_names = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
                   "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
    
    for requests in forecast_requests:
        clients_per_month = requests * conversion_rate
        lambda_hour = clients_per_month / (work_hours * days_in_month)
        n_opt, metrics = find_optimal_n(lambda_hour, mu)
        
        results.append({
            "month": f"{month_names[current_dt.month - 1]} {current_dt.year}",
            "forecast_requests": round(requests),
            "lambda_per_hour": round(lambda_hour, 4),
            "optimal_boxes": n_opt,
            "avg_wait_minutes": round(metrics["Wq"] * 60, 2) if metrics else None,
            "utilization_percent": round(metrics["rho"] * 100, 1) if metrics else None
        })
        
        if current_dt.month == 12:
            current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
            current_dt = current_dt.replace(month=current_dt.month + 1)
    
    return {"status": "success", "results": results}



# В самом конце main.py добавь (если запускаешь как скрипт):
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)