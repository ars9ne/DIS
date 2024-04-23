import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных из файла
file_path = 'pca_results.csv'  # Убедитесь, что указываете правильный путь к файлу
pca_df = pd.read_csv(file_path)

# Выведем первые несколько строк, чтобы проверить содержимое файла
print(pca_df.head())

# Визуализация данных: Scatter Plot для PC1 и PC2
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title('Scatter Plot of PC1 vs PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Добавление номера строки к каждой точке на графике
for i, txt in enumerate(pca_df.index):
    plt.text(pca_df['PC1'][i], pca_df['PC2'][i], str(txt+1))

plt.show()

# Визуализация данных: Heatmap корреляции компонентов PCA
plt.figure(figsize=(10, 8))
sns.heatmap(pca_df.corr(), annot=True, cmap='coolwarm', fmt=".10f")
plt.title('Heatmap of PCA Components Correlation')
plt.show()
