import matplotlib.pyplot as plt
from mmn_model import mmn_metrics
from optimization import find_optimal_n


def plot_dashboard(history, forecast_df, mu):

    fig, axes = plt.subplots(3, 1, figsize=(10, 14))

    # 1. История + прогноз
    axes[0].plot(history.index, history['requests'], marker='o', label="История")
    axes[0].plot(forecast_df['date'], forecast_df['forecast_requests'],
                 marker='o', label="Прогноз")

    axes[0].set_title("Исторические данные и прогноз спроса")
    axes[0].set_xlabel("Дата")
    axes[0].set_ylabel("Количество запросов")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Время ожидания
    first_lambda = forecast_df["lambda"].iloc[0]

    ns = []
    waits = []

    for n in range(1, 10):
        try:
            res = mmn_metrics(first_lambda, mu, n)
            ns.append(n)
            waits.append(res["Wq"] * 60)
        except:
            continue

    axes[1].plot(ns, waits, marker='o')
    axes[1].set_title("Ожидание vs число боксов")
    axes[1].set_xlabel("Количество боксов")
    axes[1].set_ylabel("Среднее ожидание (мин)")
    axes[1].grid(True)

    # 3. Оптимальное число боксов
    dates = []
    optimal_ns = []

    for index, row in forecast_df.iterrows():
        n, _ = find_optimal_n(row["lambda"], mu)
        if n is not None:
            dates.append(row["date"])
            optimal_ns.append(n)

    axes[2].plot(dates, optimal_ns, marker='o')
    axes[2].set_title("Динамика оптимального числа боксов")
    axes[2].set_xlabel("Дата")
    axes[2].set_ylabel("Оптимальное n")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()