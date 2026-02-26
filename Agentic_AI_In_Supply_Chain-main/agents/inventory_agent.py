import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "inventory_training.csv")
CURRENT_PATH = os.path.join(BASE_DIR, "data", "inventory.csv")


def inventory_decision(predicted_demand, supplier_reliability=None):

    train = pd.read_csv(TRAIN_PATH)

    X = train[[
        "predicted_demand",
        "current_stock",
        "past_delay",
        "holding_cost",
        "lead_time"
    ]]

    y = train["reorder_qty"]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    current = pd.read_csv(CURRENT_PATH)

    current_stock = current["current_stock"][0]
    past_delay = current["past_delay"][0]
    holding_cost = current["holding_cost"][0]
    lead_time = current["lead_time"][0]

    reorder = model.predict([[
        predicted_demand,
        current_stock,
        past_delay,
        holding_cost,
        lead_time
    ]])[0]

    reorder = int(reorder)

    # Adjust reorder based on supplier reliability
    if supplier_reliability is not None:
        if supplier_reliability < 0.6:
            reorder += 50
            print("Low reliability supplier — reorder buffer added: +50")
        elif supplier_reliability < 0.8:
            reorder += 20
            print("Medium reliability supplier — reorder buffer added: +20")

    return "Reorder", reorder