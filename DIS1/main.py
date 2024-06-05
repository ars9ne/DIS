import pandas as pd
from collections import defaultdict
import numpy as np
from itertools import combinations
file_path = 'transactions_by_dept.csv'
data = pd.read_csv(file_path)
transactions = data.groupby('POS Txn')['Dept'].apply(list).tolist()
one_hot = defaultdict(lambda: np.zeros(len(transactions)))
for i, transaction in enumerate(transactions):
    for item in transaction:
        one_hot[item][i] = 1

df_one_hot = pd.DataFrame(one_hot)


def apriori(df, min_support):
    itemset = set()
    support = {}


    for col in df.columns:
        sup = df[col].mean()
        if sup >= min_support:
            itemset.add(frozenset([col]))
            support[frozenset([col])] = sup

    k = 2
    while True:
        new_itemset = set()
        current_itemset = list(itemset)
        for i in range(len(current_itemset)):
            for j in range(i + 1, len(current_itemset)):
                union_set = current_itemset[i] | current_itemset[j]
                if len(union_set) == k:
                    sup = (df[list(union_set)].all(axis=1)).mean()
                    if sup >= min_support:
                        new_itemset.add(union_set)
                        support[union_set] = sup
        if not new_itemset:
            break
        itemset = itemset.union(new_itemset)
        k += 1

    return support



min_support = 0.005
frequent_itemsets = apriori(df_one_hot, min_support)


def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for antecedent in combinations(itemset, len(itemset) - 1):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules



min_confidence = 0.4
rules = generate_rules(frequent_itemsets, min_confidence)


def calculate_metrics(rules, frequent_itemsets, df):
    metrics = []
    total_transactions = len(df)

    for antecedent, consequent, confidence in rules:
        itemset = antecedent | consequent
        support_itemset = frequent_itemsets[itemset]
        support_antecedent = frequent_itemsets[antecedent]
        support_consequent = frequent_itemsets[consequent]

        conviction = (1 - support_consequent) / (1 - confidence)

        lift = confidence / support_consequent

        leverage = support_itemset - (support_antecedent * support_consequent)

        metrics.append((antecedent, consequent, support_itemset, confidence, conviction, lift, leverage))

    return metrics



metrics = calculate_metrics(rules, frequent_itemsets, df_one_hot)


metrics_sorted = sorted(metrics, key=lambda x: x[2], reverse=True)

for metric in metrics_sorted:
    antecedent, consequent, support_itemset, confidence, conviction, lift, leverage = metric
    print(f"Правило: {set(antecedent)} -> {set(consequent)}")
    print(f"  Поддержка: {support_itemset:.4f}")
    print(f"  Достоверность: {confidence:.4f}")
    print(f"  Уьеждённость: {conviction:.4f}")
    print(f"  Lift: {lift:.4f}")
    print(f"  Рычаг: {leverage:.4f}")
    print()
