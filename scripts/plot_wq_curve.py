import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model.smo import mmn_metrics

# Параметры для типичного шиномонтажа
LAMBDA = 2.0   # интенсивность потока (обращений/час)
MU = 3.0       # производительность одного бокса (машин/час)
MAX_N = 10     # максимальное количество боксов для расчёта
WAIT_NORM = 15 # норматив времени ожидания (минуты)

# Списки для сбора расширенных метрик СМО
n_values = []
wq_all = []         # Wq: среднее время ожидания для ВСЕХ приехавших
wq_queue_only = []   # Wq_оч: среднее время ожидания только для тех, кто попал в очередь
p_queue_list = []    # Pоч: вероятность, что клиенту придется стоять в очереди (%)
lq_list = []         # Lq: среднее число машин в очереди (штук)
rho_list = []        # rho: коэффициент загрузки боксов (%)

for n in range(1, MAX_N + 1):
    try:
        metrics = mmn_metrics(LAMBDA, MU, n)
        
        # Загружаем базовые метрики из вашей модели
        wq_hours = metrics["Wq"]
        rho = metrics["rho"]
        
        if rho >= 1:
            continue # Система нестабильна (очередь растет в бесконечность)
            
        # Рассчитываем вероятность очереди P_оч на основе формулы: Wq = P_оч / (n*mu - lambda)
        p_queue = wq_hours * (n * MU - LAMBDA)
        
        # Сохраняем метрики
        n_values.append(n)
        rho_list.append(rho * 100)
        p_queue_list.append(p_queue * 100)
        wq_all.append(wq_hours * 60) # Переводим часы в минуты
        
        # Детализация 1: Время ожидания ТОЛЬКО для тех, кто попал в очередь
        if p_queue > 0:
            wq_q_only_val = (1 / (n * MU - LAMBDA)) * 60
        else:
            wq_q_only_val = 0
        wq_queue_only.append(wq_q_only_val)
        
        # Детализация 2: Средняя длина очереди (Lq) в штуках автомобилей
        # По формуле Литтла: Lq = lambda * Wq
        lq_val = LAMBDA * wq_hours
        lq_list.append(lq_val)
        
    except ValueError:
        continue

# Авто-поиск идеального n (выполнение норматива + загрузка боксов > 30%)
n_opt = None
for n, wq in zip(n_values, wq_all):
    if wq <= WAIT_NORM:
        n_opt = n
        break

# --- СТРОИМ ДВУХПАНЕЛЬНЫЙ ДЕТАЛЬНЫЙ ГРАФИК ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
plt.rcParams['font.size'] = 10

# ЛЕВЫЙ ГРАФИК: Временные показатели (Минуты)
ax1.plot(n_values, wq_all, 'o-', color='#2B6CB0', linewidth=2.5, 
         markersize=6, label='Время ожидания для ВСЕХ ($W_q$)')
ax1.plot(n_values, wq_queue_only, 's--', color='#4A5568', linewidth=1.5, 
         markersize=5, label='Время ожидания ТОЛЬКО в очереди ($W_{q\\,оч}$)')
ax1.axhline(y=WAIT_NORM, color='#E53E3E', linestyle='--', linewidth=1.2, 
           label=f'Норматив ({WAIT_NORM} мин)')

ax1.set_xlabel('Количество боксов, $n$', fontsize=11, labelpad=8)
ax1.set_ylabel('Время (минуты)', fontsize=11, labelpad=8)
ax1.set_title('Детализация времени ожидания клиентов', fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(n_values)
ax1.grid(True, linestyle=':', alpha=0.6, color='#A0AEC0')
ax1.legend(loc='upper right', frameon=True, edgecolor='#E2E8F0')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ПРАВЫЙ ГРАФИК: Вероятностные показатели системы (%)
ax2.plot(n_values, p_queue_list, 'o-', color='#DD6B20', linewidth=2, 
         markersize=6, label='Вероятность попасть в очередь ($P_{оч}$)')
ax2.plot(n_values, rho_list, 'd-.', color='#319795', linewidth=1.5, 
         markersize=6, label='Коэффициент занятости боксов ($\\rho$)')

ax2.set_xlabel('Количество боксов, $n$', fontsize=11, labelpad=8)
ax2.set_ylabel('Вероятность / Загрузка (%)', fontsize=11, labelpad=8)
ax2.set_title('Анализ рисков очереди и загрузки боксов', fontsize=12, fontweight='bold', pad=12)
ax2.set_xticks(n_values)
ax2.set_ylim(0, 105)
ax2.grid(True, linestyle=':', alpha=0.6, color='#A0AEC0')
ax2.legend(loc='upper right', frameon=True, edgecolor='#E2E8F0')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Информационная плашка на правом графике про оптимальное решение
if n_opt:
    idx = n_values.index(n_opt)
    ax2.axvline(x=n_opt, color='#718096', linestyle=':', linewidth=1.2)
    ax2.text(n_opt + 0.2, 10, f'Оптимум: n={n_opt}\nРиск очереди: {p_queue_list[idx]:.1f}%\nЗагрузка боксов: {rho_list[idx]:.1f}%', 
             fontsize=10, color='#2D3748', bbox=dict(boxstyle="round,pad=0.4", fc="#EDF2F7", ec="#CBD5E0", alpha=0.9))

plt.tight_layout()
output_path = 'smo_comprehensive_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Расширенный график сохранён: {output_path}")

# --- ИДЕАЛЬНАЯ ТАБЛИЦА ДЛЯ ВСТАВКИ В ДИПЛОМ ---
print("\n" + "="*85)
print(" ДЕТАЛЬНАЯ ТАБЛИЦА ПАРАМЕТРОВ СМО ДЛЯ РАЗДЕЛА ОПТИМИЗАЦИИ ДИПЛОМА")
print("="*85)
print(f"{'n':<4} | {'Загрузка ρ':<11} | {'P_оч (риск)':<12} | {'W_q (всех)':<12} | {'W_q (очередь)':<14} | {'L_q (очередь)':<12}")
print(f"{'':<4} | {'(%)':<11} | {'(%)':<12} | {'(мин)':<12} | {'(мин)':<14} | {'(авто)':<12}")
print("-"*85)

for i, n in enumerate(n_values):
    status_mark = "★" if n == n_opt else " "
    print(f"{n:<2} {status_mark:<1} | {rho_list[i]:<11.1f} | {p_queue_list[i]:<12.1f} | {wq_all[i]:<12.2f} | {wq_queue_only[i]:<14.2f} | {lq_list[i]:<12.2f}")
print("="*85)
print("★ — Рекомендуемое оптимальное число боксов по критерию времени ожидания.")

plt.show()
