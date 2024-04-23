import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
#pd options
pd.set_option('display.max_columns', None)  # Показывать все столбцы
pd.set_option('display.max_rows', None)     # Показывать все строки
pd.set_option('display.width', 1000)        # Установить ширину для отображения
pd.set_option('display.max_colwidth', None) # Показывать полную длину содержимого столбца

# Загрузка данных
file_path = 'auto-mpg.data'
column_names = [
    'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
    'acceleration', 'model_year', 'origin', 'car_name'
]
data = pd.read_csv(file_path, delim_whitespace=True, names=column_names)
# Преобразование 'horsepower' в числовой формат
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
# Удаление строк с пропущенными значениями
data.dropna(inplace=True)
# Удаление дубликатов
data.drop_duplicates(inplace=True)

# Step 2: Feature Selection
# We'll use PCA to understand the importance of each feature
# First, we need to isolate numeric columns for PCA
numeric_features = data.select_dtypes(include=[np.number])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)

# Applying PCA
pca = PCA(n_components=0.95)  # retain 95% of variance
principal_components = pca.fit_transform(scaled_features)
explained_variance_ratio = pca.explained_variance_ratio_

# Create a DataFrame from the PCA results
pca_columns = ['PC' + str(i+1) for i in range(len(explained_variance_ratio))]
pca_df = pd.DataFrame(principal_components, columns=pca_columns)

# Include car names in PCA DataFrame for export
pca_df_with_names = pca_df.copy()
pca_df_with_names['car_name'] = data['car_name']
pca_df_with_names['model_year'] = data['model_year']

# Save PCA results to files
pca_df.to_csv('pca_results.csv', index=False)
pca_df_with_names.to_csv('pca_results_with_names.csv', index=False)

# Loadings and variance explanation
loadings = pca.components_
loadings_df = pd.DataFrame(loadings.T, columns=pca_columns, index=numeric_features.columns)
print("Loadings:\n", loadings_df)
print("Explained variance ratio:", explained_variance_ratio)
