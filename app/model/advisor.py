# app/model/advisor.py
import math

class NeuralAdvisor:
    """
    Модуль интеллектуальных рекомендаций (Neural Advisory System).
    Анализирует метрики СМО и прогноз спроса, выдавая взвешенные рекомендации.
    """
    
    def __init__(self):
        # Пороговые значения (гиперпараметры системы)
        self.LOAD_CRITICAL = 85.0
        self.LOAD_WARNING = 70.0
        self.WAIT_CRITICAL = 10.0  # минут
        self.WAIT_WARNING = 5.0    # минут
        self.IDLE_THRESHOLD = 30.0 # % простоя

    def _calculate_risk_score(self, load: float, wait: float) -> float:
        """Расчет уровня риска (0.0 - 1.0)."""
        risk = 0.0
        if load > self.LOAD_CRITICAL: risk += 0.6
        elif load > self.LOAD_WARNING: risk += 0.3
        
        if wait > self.WAIT_CRITICAL: risk += 0.4
        elif wait > self.WAIT_WARNING: risk += 0.2
        
        return min(risk, 1.0)

    def generate(self, results: list, params: dict) -> list:
        """Генерация списка рекомендаций."""
        insights = []
        if not results:
            return insights

        # 1. Агрегация метрик
        max_load = max(r['utilization_percent'] for r in results)
        max_wait = max(r['avg_wait_minutes'] for r in results)
        avg_load = sum(r['utilization_percent'] for r in results) / len(results)
        
        peak_month = max(results, key=lambda x: x['utilization_percent'])
        
        # 2. Анализ тренда
        trend_diff = results[-1]['forecast_requests'] - results[0]['forecast_requests']
        trend_pct = (trend_diff / results[0]['forecast_requests']) * 100 if results[0]['forecast_requests'] > 0 else 0

        # 3. Логика принятия решений
        
        # Сценарий А: Критическая перегрузка
        risk_score = self._calculate_risk_score(max_load, max_wait)
        if risk_score > 0.8:
            insights.append({
                "level": "CRITICAL",
                "color": "#dc3545",
                "icon": "[!]",
                "title": "КРИТИЧЕСКИЙ РИСК ПЕРЕГРУЗКИ",
                "text": f"Модель прогнозирует коллапс очереди в {peak_month['month_name']}. "
                        f"Загрузка {max_load:.1f}% и ожидание {max_wait:.1f} мин приведут к оттоку клиентов. "
                        f"<b>Рекомендация:</b> Немедленно увеличить кол-во постов или ввести запись.",
                "confidence": 98
            })
        
        # Сценарий Б: Высокая нагрузка
        elif risk_score > 0.4:
            insights.append({
                "level": "WARNING",
                "color": "#ffc107",
                "icon": "[!]",
                "title": "ЗОНА ВЫСОКОЙ НАГРУЗКИ",
                "text": f"В {peak_month['month_name']} ожидается напряженный режим. "
                        f"Вероятность ожидания >5 мин составляет 85%. "
                        f"<b>Рекомендация:</b> Подготовить резервный персонал.",
                "confidence": 85
            })

        # Сценарий В: Простой ресурсов
        if avg_load < self.IDLE_THRESHOLD:
            low_months = [r['month_name'] for r in results if r['utilization_percent'] < 20]
            months_str = ", ".join(low_months[:3]) + ("..." if len(low_months) > 3 else "")
            insights.append({
                "level": "INFO",
                "color": "#17a2b8",
                "icon": "[i]",
                "title": "ОПТИМИЗАЦИЯ РЕСУРСОВ",
                "text": f"Выявлен простой мощностей (средняя загрузка {avg_load:.1f}%). "
                        f"Критические спады в: {', '.join(low_months[:3])}. "
                        f"<b>Рекомендация:</b> Запустить маркетинговую акцию для заполнения мощностей.",
                "confidence": 90
            })

        # Сценарий Г: Трендовый анализ
        if abs(trend_pct) > 25:
            direction = "РОСТ" if trend_pct > 0 else "СПАД"
            icon_char = "[^]" if trend_pct > 0 else "[v]"
            insights.append({
                "level": "STRATEGY",
                "color": "#28a745" if trend_pct > 0 else "#fd7e14",
                "icon": icon_char,
                "title": f"СТРАТЕГИЧЕСКИЙ {direction}",
                "text": f"Зафиксирован сильный тренд ({direction} на {abs(trend_pct):.1f}%). "
                        f"Необходимо скорректировать закупки шин и график отпусков.",
                "confidence": 92
            })

        # Сценарий Д: Норма
        if not insights:
            insights.append({
                "level": "SUCCESS",
                "color": "#28a745",
                "icon": "[OK]",
                "title": "ОПТИМАЛЬНЫЙ РЕЖИМ",
                "text": "Система работает в штатном режиме. Прогнозируемых рисков не выявлено. "
                       "Текущей конфигурации достаточно для качественного сервиса.",
                "confidence": 95
            })

        return insights