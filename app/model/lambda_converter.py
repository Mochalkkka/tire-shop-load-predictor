def convert_requests_to_lambda(forecast_df, conversion_rate=0.1, work_hours=10, days_in_month=30):
    lambdas = []
    for requests in forecast_df['forecast_requests']:
        clients = requests * conversion_rate
        hourly_lambda = clients / (work_hours * days_in_month)
        lambdas.append(hourly_lambda)
    
    forecast_df['lambda'] = lambdas
    return forecast_df