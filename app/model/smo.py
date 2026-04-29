# app/model/smo.py
import math

def mmn_metrics(lmbda: float, mu: float, n: int) -> dict:
    """Расчёт метрик для M/M/n"""
    if lmbda >= n * mu:
        raise ValueError(f"Система неустойчива: λ={lmbda:.3f} >= n·μ={n*mu}")
    
    rho = lmbda / (n * mu)
    
    sum1 = sum((lmbda/mu)**k / math.factorial(k) for k in range(n))
    sum2 = ((lmbda/mu)**n / math.factorial(n)) * (1/(1-rho))
    P0 = 1 / (sum1 + sum2)
    
    Lq = (P0 * (lmbda/mu)**n * rho) / (math.factorial(n) * (1-rho)**2)
    Wq = Lq / lmbda if lmbda > 0 else 0
    W = Wq + 1/mu
    
    return {
        "rho": rho,
        "P0": P0,
        "Lq": Lq,
        "Wq": Wq,
        "W": W
    }

def find_optimal_n(lmbda: float, mu: float, max_n: int = 10, max_wait_min: float = 15) -> tuple:
    """Подбор минимального n, чтобы ожидание ≤ max_wait_min"""
    for n in range(1, max_n + 1):
        try:
            metrics = mmn_metrics(lmbda, mu, n)
            if metrics["Wq"] * 60 <= max_wait_min:
                return n, metrics
        except:
            continue
    return None, None