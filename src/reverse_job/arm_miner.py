from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class ARMMiner:
    """Association Rule Mining으로 공고 간 공동 출현 패턴을 학습한다."""

    def __init__(self, min_support: float = 0.2, min_confidence: float = 0.6) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence

    def mine(self, transactions: list[list[str]]) -> pd.DataFrame:
        if not transactions:
            return pd.DataFrame()

        encoder = TransactionEncoder()
        encoded = encoder.fit(transactions).transform(transactions)
        df = pd.DataFrame(encoded, columns=encoder.columns_)

        freq = apriori(df, min_support=self.min_support, use_colnames=True)
        if freq.empty:
            return pd.DataFrame()

        return association_rules(
            freq, metric="confidence", min_threshold=self.min_confidence
        )

    @staticmethod
    def find_implicit(target_skills: set[str], rules: pd.DataFrame) -> list[str]:
        if rules.empty:
            return []

        implicit: set[str] = set()
        for _, row in rules.iterrows():
            antecedents = set(row["antecedents"])
            consequents = set(row["consequents"])
            if antecedents.issubset(target_skills):
                implicit.update(consequents - target_skills)
        return sorted(implicit)
