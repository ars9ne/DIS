import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Загрузка данных
file_path = 'Bitcoin Historical Data.csv'
data = pd.read_csv(file_path)

# Преобразование столбца 'Date' в формат datetime и установка его в качестве индекса
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.set_index('Date', inplace=True)

# Преобразование столбцов в числовой формат
for column in ['Price', 'Open', 'High', 'Low']:
    data[column] = data[column].str.replace(',', '').astype(float)

data['Change %'] = data['Change %'].str.replace('%', '').astype(float)

# Установка частоты индекса на недельную
data = data.asfreq('W')

# Определение и обучение модели SARIMAX
model = SARIMAX(data['Price'],
                order=(1, 1, 1),
                seasonal_order=(0, 1, 0, 208),
                enforce_stationarity=False,
                enforce_invertibility=False)

sarimax_model = model.fit(disp=False)

# Вывод результата
print(sarimax_model.summary())
# Визуализация остатков модели
residuals = sarimax_model.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Остатки модели SARIMAX')
plt.show()

# Автокорреляция остатков
plot_acf(residuals)
plt.show()

# Прогнозирование на следующие 52 недели (1 год)
forecast = sarimax_model.get_forecast(steps=52)
forecast_index = pd.date_range(start=data.index[-1], periods=52, freq='W')
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(data['Price'], label='Исторические данные')
plt.plot(forecast_index, forecast_values, label='Прогноз', color='red')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Прогноз цены биткоина на следующие 52 недели')
plt.legend()
plt.show()