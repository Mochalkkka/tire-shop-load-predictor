from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from datetime import date
import os

from app.model.ensemble import EnsemblePredictor
from app.model.smo import find_optimal_n

app = FastAPI(title="🔧 Шиномонтаж Прогноз")

# 🔹 Шаблоны
templates = Jinja2Templates(directory="app/templates")

# 🔹 Загрузка моделей
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pth")
LINEAR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "linear_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Файл модели не найден: {MODEL_PATH}")
if not os.path.exists(LINEAR_PATH):
    raise RuntimeError(f"❌ Файл линейной модели не найден: {LINEAR_PATH}")

predictor = EnsemblePredictor(MODEL_PATH, LINEAR_PATH)

# 🔹 Главная страница
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

# 🔹 Страница с результатами
@app.get("/predict")
def predict_page(
    request: Request,
    boxes: int = 3,
    mu: float = 3.0,
    start_month: int = 0,
    months: int = 6,
    conversion: float = 10.0,
    work_hours: int = 10
):
    # Прогноз
    forecast_requests = predictor.forecast(steps=12)
    forecast_requests = forecast_requests[start_month:start_month + months]
    
    # Расчёт СМО
    results = []
    month_names = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
                   "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
    
    current_month_idx = start_month
    days_in_month = 30
    
    for i, requests in enumerate(forecast_requests):
        clients_per_month = requests * (conversion / 100)
        lambda_hour = clients_per_month / (work_hours * days_in_month)
        n_opt, metrics = find_optimal_n(lambda_hour, mu)
        
        results.append({
            "month_name": month_names[current_month_idx],
            "forecast_requests": round(requests),
            "lambda_per_hour": round(lambda_hour, 4),
            "optimal_boxes": n_opt if n_opt else "-",
            "avg_wait_minutes": round(metrics["Wq"] * 60, 2) if metrics else "-",
            "utilization_percent": round(metrics["rho"] * 100, 1) if metrics else 0
        })
        current_month_idx = (current_month_idx + 1) % 12
    
    # ✅ ИСПРАВЛЕННЫЙ возврат шаблона:
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "results": results,
            "params": {
                "boxes": boxes,
                "mu": mu,
                "months": months,
                "work_hours": work_hours,
                "conversion": conversion
            }
        }
    )

# 🔹 API endpoint (без изменений)
@app.get("/api/predict")
def predict_api(
    boxes: int = 3,
    mu: float = 3.0,
    months: int = 12,
    conversion_rate: float = 0.1,
    work_hours: int = 10,
    days_in_month: int = 30
):
    forecast_requests = predictor.forecast(steps=months)
    
    results = []
    for i, requests in enumerate(forecast_requests):
        clients_per_month = requests * conversion_rate
        lambda_hour = clients_per_month / (work_hours * days_in_month)
        n_opt, metrics = find_optimal_n(lambda_hour, mu)
        
        results.append({
            "month_offset": i + 1,
            "forecast_requests": round(requests),
            "lambda_per_hour": round(lambda_hour, 4),
            "optimal_boxes": n_opt,
            "avg_wait_minutes": round(metrics["Wq"] * 60, 2) if metrics else None,
            "utilization_percent": round(metrics["rho"] * 100, 1) if metrics else None
        })
    
    return {
        "status": "success",
        "parameters": {
            "boxes": boxes,
            "mu": mu,
            "conversion_rate": conversion_rate,
            "work_hours": work_hours
        },
        "results": results
    }