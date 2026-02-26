import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "supplier_training.csv")
SUPPLIER_PATH = os.path.join(BASE_DIR, "data", "suppliers.csv")


def select_supplier(reorder):

    train = pd.read_csv(TRAIN_PATH)

    X = train[[
        "cost",
        "delivery_time",
        "past_delays",
        "quality_score"
    ]]

    y = train["on_time_delivery"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    suppliers = pd.read_csv(SUPPLIER_PATH)

    suppliers["predicted_score"] = model.predict_proba(
        suppliers[[
            "cost",
            "delivery_time",
            "past_delays",
            "quality_score"
        ]]
    )[:, 1]

    # Adjust selection logic based on reorder quantity
    if reorder > 300:
        suppliers = suppliers.sort_values(
            by="predicted_score",
            ascending=False
        )
    else:
        suppliers = suppliers.sort_values(
            by=["predicted_score", "cost"],
            ascending=[False, True]
        )

    best = suppliers.iloc[0]

    # ✅ Return supplier name, reliability float, AND full dataframe
    reliability = float(best["reliability"])
    return best["supplier"], reliability, suppliers