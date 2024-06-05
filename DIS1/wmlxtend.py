import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Загрузка данных
file_path = 'transactions_by_dept.csv'
data = pd.read_csv(file_path)

# Группируем данные по идентификатору транзакции и создаем списки товаров для каждой транзакции
transactions = data.groupby('POS Txn')['Dept'].apply(list).tolist()

# Преобразуем список транзакций в формат one-hot encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Найдем частые наборы элементов с помощью алгоритма Apriori
min_support = 0.005  # Минимальная поддержка
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

# Генерируем ассоциативные правила с минимальной достоверностью 0.4
min_confidence = 0.4  # Минимальная достоверность
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Рассчитаем дополнительные метрики (лифт, рычаг и убежденность)
rules["conviction"] = (1 - rules["consequent support"]) / (1 - rules["confidence"])

# Сортируем правила по поддержке от большего к меньшему
rules = rules.sort_values(by="support", ascending=False)

# Вывод метрик
for _, row in rules.iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    print(f"Rule: {{{antecedents}}} -> {{{consequents}}}")
    print(f"  Support: {row['support']:.4f}")
    print(f"  Confidence: {row['confidence']:.4f}")
    print(f"  Conviction: {row['conviction']:.4f}")
    print(f"  Lift: {row['lift']:.4f}")
    print(f"  Leverage: {row['leverage']:.4f}")
    print()
