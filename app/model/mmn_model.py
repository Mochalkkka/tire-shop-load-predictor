import math

def mmn_metrics(lmbda, mu, n):
    if lmbda >= n * mu:
        raise ValueError("Система неустойчива: λ >= n·μ")
    
    rho = lmbda / (n * mu)
    
    sum1 = sum((lmbda / mu) ** k / math.factorial(k) for k in range(n))
    sum2 = ((lmbda / mu) ** n / math.factorial(n)) * (1 / (1 - rho))
    
    P0 = 1 / (sum1 + sum2)
    Lq = (P0 * (lmbda / mu) ** n * rho) / (math.factorial(n) * (1 - rho) ** 2)
    Wq = Lq / lmbda
    W = Wq + 1 / mu
    
    return {"rho": rho, "P0": P0, "Lq": Lq, "Wq": Wq, "W": W}