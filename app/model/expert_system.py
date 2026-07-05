# app/model/expert_system.py
"""
Модуль экспертной системы поддержки принятия решений (DSS).
Реализует бизнес-логику анализа метрик СМО и выработки рекомендаций
на основе отраслевых нормативов и эвристических правил.
"""

class TireServiceExpertSystem:
    """
    Экспертная система для анализа загрузки шиномонтажного сервиса.
    Использует детерминированные правила для интерпретации результатов моделирования M/M/n.
    """

    # Отраслевые нормативы и пороговые значения (константы предметной области)
    MAX_ACCEPTABLE_WAIT_TIME_MIN = 15.0  # Макс. время ожидания по стандарту качества
    LOAD_WARNING_THRESHOLD = 0.70        # 70% загрузки - зона внимания
    LOAD_CRITICAL_THRESHOLD = 0.85       # 85% загрузки - риск коллапса очереди
    IDLE_THRESHOLD = 0.30                # 30% загрузки - неэффективное использование

    def __init__(self):
        pass

    def analyze(self, forecast_results: list, config: dict) -> list:
        """
        Основной метод анализа. Принимает результаты прогноза и конфигурацию сервиса.
        Возвращает список структурированных заключений эксперта.
        
        Args:
            forecast_results: Список словарей с метриками за каждый месяц.
            config: Параметры сервиса (боксы, производительность и т.д.).
            
        Returns:
            Список словарей вида: {type, severity, title, description, action}
        """
        conclusions = []
        
        if not forecast_results:
            return conclusions

        # 1. Агрегация ключевых показателей за период
        max_utilization = max(r['utilization_percent'] for r in forecast_results) / 100.0
        max_wait_time = max(r['avg_wait_minutes'] for r in forecast_results)
        avg_utilization = sum(r['utilization_percent'] for r in forecast_results) / len(forecast_results) / 100.0
        
        # Поиск месяца с пиковой нагрузкой
        peak_month_data = max(forecast_results, key=lambda x: x['utilization_percent'])
        peak_month_name = peak_month_data['month_name']

        # 2. Применение правил вывода (Rule-Based Inference)

        # Правило 1: Критическая перегрузка (Риск потери клиентов)
        if max_utilization >= self.LOAD_CRITICAL_THRESHOLD:
            conclusions.append({
                "type": "risk",
                "severity": "high",
                "title": f"Критическая перегрузка в {peak_month_name}",
                "description": (
                    f"Прогнозируемая загрузка каналов обслуживания достигает {max_utilization*100:.1f}%. "
                    f"Время ожидания в очереди может превысить {max_wait_time:.1f} мин, "
                    f"что критически выше норматива ({self.MAX_ACCEPTABLE_WAIT_TIME_MIN} мин). "
                    f"Высока вероятность оттока клиентов."
                ),
                "action": (
                    f"Рекомендуется увеличить количество рабочих постов минимум на 1 единицу "
                    f"или внедрить систему предварительной записи для сглаживания пика."
                )
            })
        
        # Правило 2: Зона повышенной нагрузки (Предупреждение)
        elif max_utilization >= self.LOAD_WARNING_THRESHOLD:
            conclusions.append({
                "type": "warning",
                "severity": "medium",
                "title": f"Высокая нагрузка в {peak_month_name}",
                "description": (
                    f"Загрузка системы приближается к предельной ({max_utilization*100:.1f}%). "
                    f"Вероятность образования очередей возрастает экспоненциально."
                ),
                "action": (
                    "Рекомендуется подготовить резервный персонал или оптимизировать "
                    "технологический процесс (увеличить μ) для снижения времени обслуживания."
                )
            })

        # Правило 3: Неэффективное использование ресурсов (Простой)
        if avg_utilization < self.IDLE_THRESHOLD:
            low_load_months = [r['month_name'] for r in forecast_results if r['utilization_percent'] < 20.0]
            periods_str = ", ".join(low_load_months[:3])
            
            conclusions.append({
                "type": "optimization",
                "severity": "low",
                "title": "Низкая эффективность использования мощностей",
                "description": (
                    f"Средняя загрузка сервиса за период составляет всего {avg_utilization*100:.1f}%. "
                    f"Наблюдается значительный простой ресурсов в месяцы: {periods_str}."
                ),
                "action": (
                    "Целесообразно рассмотреть возможность сокращения количества активных постов "
                    "в низкий сезон или запуска маркетинговых акций (скидки на хранение, диагностику) "
                    "для стимулирования спроса."
                )
            })

        # Правило 4: Анализ тренда спроса (Стратегия)
        if len(forecast_results) > 1:
            start_demand = forecast_results[0]['forecast_requests']
            end_demand = forecast_results[-1]['forecast_requests']
            trend_coeff = (end_demand - start_demand) / start_demand if start_demand > 0 else 0
            
            if abs(trend_coeff) > 0.25: # Изменение более чем на 25%
                direction = "роста" if trend_coeff > 0 else "падения"
                conclusions.append({
                    "type": "strategy",
                    "severity": "info",
                    "title": f"Выявлен тренд {direction} спроса",
                    "description": (
                        f"За прогнозируемый период спрос изменяется на {abs(trend_coeff)*100:.1f}%. "
                        f"Требуется корректировка операционного плана."
                    ),
                    "action": (
                        "Чтобы избежать простоев в пик сезона, рекомендуется заранее увеличить складские запасы шин "
                        "и расходников, а также провести техническое обслуживание оборудования до начала высокой нагрузки."
                    )
                })

        # Если критических проблем нет
        if not conclusions:
            conclusions.append({
                "type": "success",
                "severity": "none",
                "title": "Штатный режим работы",
                "description": (
                    f"Прогнозируемые параметры нагрузки находятся в оптимальном диапазоне. "
                    f"Текущая конфигурация сервиса ({config.get('boxes', 0)} боксов) полностью "
                    f"удовлетворяет спросу с соблюдением нормативов времени ожидания."
                ),
                "action": "Дополнительных вмешательств не требуется."
            })

        return conclusions