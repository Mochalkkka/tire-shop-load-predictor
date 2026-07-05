# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from fastapi.responses import StreamingResponse
from datetime import datetime
import os
import numpy as np
import pandas as pd

# Импорт моделей и модулей
from app.model.predictor import ModelPredictor
from app.model.smo import find_optimal_n
from app.model.linear_model import LinearTrendModel
from app.model.ensemble import EnsemblePredictor
from app.model.expert_system import TireServiceExpertSystem

app = FastAPI(title="Прогноз загрузки шиномонтажа")

# ==============================================================================
# 🔹 1. НАСТРОЙКА ПРИЛОЖЕНИЯ
# ==============================================================================

# Подключение статических файлов (графиков, стилей)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Настройка шаблонов Jinja2
templates = Jinja2Templates(directory="app/templates")

# ==============================================================================
# 🔹 2. ЗАГРУЗКА МОДЕЛЕЙ (Выполняется один раз при старте сервера)
# ==============================================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pth")
LINEAR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "linear_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Файл модели не найден: {MODEL_PATH}\nЗапусти: python scripts/train.py")

# Инициализация предсказателей
predictor = ModelPredictor(MODEL_PATH)
lr_model = LinearTrendModel.load(LINEAR_PATH)
ensemble_model = EnsemblePredictor(MODEL_PATH, LINEAR_PATH)

# Инициализация экспертной системы (бизнес-логика)
expert_system = TireServiceExpertSystem()

# ==============================================================================
# 🔹 3. МАРШРУТЫ (ROUTES)
# ==============================================================================

@app.get("/")
def home(request: Request):
    """Главная страница с формой ввода"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@app.get("/predict")
def predict_page(
    request: Request,
    boxes: int = 3,
    mu: float = 3.0,
    start_date: str = None,
    end_date: str = None,
    conversion: float = 10.0,
    work_hours: int = 10,
    model_type: str = "nn_seasonal" 
):
    """Страница результатов прогноза и заключений экспертной системы"""
    
    # 1. Валидация входных данных
    if not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Укажите даты начала и окончания прогноза")
    
    try:
        start_dt = datetime.strptime(start_date + "-01", "%Y-%m-%d")
        end_dt = datetime.strptime(end_date + "-01", "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Неверный формат даты. Используйте YYYY-MM")
    
    if end_dt < start_dt:
        raise HTTPException(status_code=400, detail="Дата окончания должна быть позже даты начала")
    
    months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    
    if months_diff < 1:
        raise HTTPException(status_code=400, detail="Минимальный период: 1 месяц")
    if months_diff > 24:
        raise HTTPException(status_code=400, detail="Максимальный период: 24 месяца")
    
    # 2. Прогнозирование спроса (выбор модели)
    steps = months_diff + 1
    start_month = start_dt.month - 1
    
    if model_type == "nn_seasonal":
        forecast_requests = predictor.forecast(steps=steps, start_month=start_month, apply_seasonality=True)
    elif model_type == "nn_only":
        forecast_requests = predictor.forecast(steps=steps, start_month=start_month, apply_seasonality=False)
    elif model_type == "linear":
        forecast_requests = [max(0, x) for x in lr_model.predict(steps=steps)]
    elif model_type == "ma":
        history = list(predictor.scaler.inverse_transform(np.array(predictor.last_sequence).reshape(-1, 1)).flatten())
        forecast_requests = []
        for _ in range(steps):
            pred = np.mean(history[-3:]) if len(history) >= 3 else history[-1]
            forecast_requests.append(pred)
            history.append(pred)
    elif model_type == "ensemble":
        forecast_requests = ensemble_model.forecast(steps=steps)
    else:
        forecast_requests = predictor.forecast(steps=steps, start_month=start_month, apply_seasonality=True)
    
    # 3. Расчёт метрик СМО (M/M/n)
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
        
        # Переход к следующему месяцу
        if current_dt.month == 12:
            current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
            current_dt = current_dt.replace(month=current_dt.month + 1)
    
    # 4. ГЕНЕРАЦИЯ ЗАКЛЮЧЕНИЙ ЭКСПЕРТНОЙ СИСТЕМЫ
    expert_conclusions = expert_system.analyze(
        forecast_results=results, 
        config={
            "boxes": boxes,
            "mu": mu,
            "conversion": conversion
        }
    )
        
    # 5. Возврат HTML-шаблона с данными
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "request": request,
            "results": results,
            "conclusions": expert_conclusions,  # Передаем заключения эксперта
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
    """API эндпоинт для программного доступа (без HTML)"""
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

@app.get("/export-excel")
def export_excel(
    start_date: str,
    end_date: str,
    boxes: int,
    mu: float,
    conversion: float,
    work_hours: int = 10
):
    """Генерация Excel отчета с полным пересчетом данных"""
    try:
        # 1. Парсинг дат (копируем логику из predict_page)
        start_dt = datetime.strptime(start_date + "-01", "%Y-%m-%d")
        end_dt = datetime.strptime(end_date + "-01", "%Y-%m-%d")
        months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        
        if months_diff < 1 or months_diff > 24:
            raise ValueError("Период должен быть от 1 до 24 месяцев")

        # 2. Получаем прогноз от модели (используем тот же predictor)
        # Берем базовую модель (nn_seasonal), так как экспорт обычно нужен для основного сценария
        forecast_requests = predictor.forecast(
            steps=months_diff + 1, 
            start_month=start_dt.month - 1, 
            apply_seasonality=True
        )
        
        # 3. Расчет метрик СМО (копируем логику расчета results)
        results = []  # <-- ВОТ ТУТ МЫ СОЗДАЕМ СПИСОК, КОТОРОГО НЕ ХВАТАЛО
        month_names = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
                       "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
        
        current_dt = start_dt
        days_in_month = 30
        
        for requests in forecast_requests:
            clients_per_month = requests * (conversion / 100)
            lambda_hour = clients_per_month / (work_hours * days_in_month)
            n_opt, metrics = find_optimal_n(lambda_hour, mu)
            
            results.append({
                "Месяц": f"{month_names[current_dt.month - 1]} {current_dt.year}",
                "Прогноз запросов": int(round(requests)),
                "Клиентов (мес)": int(round(clients_per_month)),
                "λ (клиентов/час)": round(lambda_hour, 4),
                "Оптимально боксов": n_opt if n_opt else 1,
                "Ожидание (мин)": round(metrics["Wq"] * 60, 2) if metrics else 0,
                "Загрузка (%)": round(metrics["rho"] * 100, 1) if metrics else 0
            })
            
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
                
    except Exception as e:
        # Если ошибка, возвращаем текстовое сообщение вместо краша сервера
        return {"error": f"Не удалось сформировать отчет: {str(e)}"}

    # 4. Создаем Excel файл в памяти
    df = pd.DataFrame(results)
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Лист 1: Детальный прогноз
            df.to_excel(writer, sheet_name='Прогноз загрузки', index=False)
            
            # Лист 2: Параметры отчета
            params_data = [
                ["Параметр", "Значение"],
                ["Дата начала", start_date],
                ["Дата окончания", end_date],
                ["Количество боксов (текущее)", boxes],
                ["Производительность (μ)", mu],
                ["Конверсия", f"{conversion}%"],
                ["Часов работы в день", work_hours],
                ["Модель прогноза", "Neural Network + Seasonality"]
            ]
            pd.DataFrame(params_data).to_excel(writer, sheet_name='Параметры', index=False)
            
            # Авто-ширина колонок (для красоты)
            worksheet = writer.sheets['Прогноз загрузки']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    except Exception as e:
        return {"error": f"Ошибка записи Excel: {str(e)}"}

    output.seek(0)
    
    # Формируем имя файла
    filename = f"report_tire_{start_date}_{end_date}.xlsx"
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Запуск для локальной разработки или деплоя
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)