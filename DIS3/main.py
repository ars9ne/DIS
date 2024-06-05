import pandas as pd

# Конвертация german.data
input_file_categorical = 'german.data'
# Определите соответствующий разделитель
delimiter_categorical = ' '  # Замените, если это не пробел

# Чтение категориального .data файла
data_categorical = pd.read_csv(input_file_categorical, delimiter=delimiter_categorical, header=None)

# Сохранение в .csv файл
output_file_categorical = 'german_categorical.csv'
data_categorical.to_csv(output_file_categorical, index=False)

print(f"Категориальный файл успешно конвертирован и сохранен как {output_file_categorical}")
