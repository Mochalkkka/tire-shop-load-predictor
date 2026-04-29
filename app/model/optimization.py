from app.model.mmn_model import mmn_metrics

def find_optimal_n(lmbda, mu, max_n=15, max_wait=15):
    for n in range(1, max_n + 1):
        try:
            res = mmn_metrics(lmbda, mu, n)
            if res["Wq"] * 60 <= max_wait:  # минуты
                return n, res
        except:
            continue
    return None, None