from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из файла
file_path = 'pca_results.csv'  # Убедитесь, что указываете правильный путь к файлу
pca_df = pd.read_csv(file_path)

# Шаг 1: Метод локтя для определения количества кластеров
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_df)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Выберите значение k на основе графика локтя
optimal_k = 3  # Пример, замените на ваше выбранное значение

# Шаг 2: Кластеризация с помощью K-средних
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(pca_df)

# Добавляем метки кластеров в исходный DataFrame
pca_df['Cluster'] = clusters

# Визуализация кластеров
sns.scatterplot(x='PC1', y='PC3', hue='Cluster', data=pca_df, palette='viridis')
plt.title(f'PCA Clusters (k={optimal_k})')
plt.show()
