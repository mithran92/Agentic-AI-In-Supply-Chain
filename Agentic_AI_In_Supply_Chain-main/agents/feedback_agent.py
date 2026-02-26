import os
import pandas as pd
from difflib import get_close_matches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PERFORMANCE_PATH = os.path.join(BASE_DIR, "data", "performance.csv")
SUPPLIER_PATH = os.path.join(BASE_DIR, "data", "suppliers.csv")


def update_reliability(selected_supplier):

    perf = pd.read_csv(PERFORMANCE_PATH)
    suppliers = pd.read_csv(SUPPLIER_PATH)

    # -----------------------------
    # 🔥 Normalize supplier names
    # -----------------------------
    suppliers["supplier"] = suppliers["supplier"].astype(str).str.strip().str.lower()
    perf["supplier"] = perf["supplier"].astype(str).str.strip().str.lower()

    selected_supplier = str(selected_supplier).strip().lower()

    # -----------------------------
    # 🔥 Update reliability scores
    # -----------------------------
    for _, row in perf.iterrows():

        supplier = row["supplier"]
        delay = row["delivery_delay"]
        quality_issue = row["quality_issue"]

        reward = 1

        if delay > 1:
            reward -= 0.5

        if quality_issue == 1:
            reward -= 0.5

        suppliers.loc[
            suppliers["supplier"] == supplier,
            "reliability"
        ] += reward * 0.02

    # Keep reliability between 0 and 1
    suppliers["reliability"] = suppliers["reliability"].clip(0, 1)

    suppliers.to_csv(SUPPLIER_PATH, index=False)

    # -----------------------------
    # 🔥 Safe retrieval of selected supplier
    # -----------------------------
    filtered = suppliers.loc[
        suppliers["supplier"] == selected_supplier,
        "reliability"
    ]

    # If exact match not found → try fuzzy match
    if filtered.empty:
        match = get_close_matches(selected_supplier, suppliers["supplier"], n=1)
        if match:
            selected_supplier = match[0]
            filtered = suppliers.loc[
                suppliers["supplier"] == selected_supplier,
                "reliability"
            ]
        else:
            print("⚠ Supplier not found:", selected_supplier)
            return None

    return float(filtered.values[0])