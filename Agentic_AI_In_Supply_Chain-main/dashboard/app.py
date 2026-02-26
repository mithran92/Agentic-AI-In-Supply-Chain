import sys
import os

# Fix path so agents can be imported
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

from llm.llm_helper import run_llm_agent


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Agentic AI Supply Chain",
    layout="wide"
)

st.title("🤖 Agentic AI Supply Chain Dashboard")
st.write("LLM-driven autonomous supply chain optimization system")


# -----------------------------
# LOAD DATA
# -----------------------------
sales_path = os.path.join(BASE_DIR, "data", "sales.csv")
suppliers_path = os.path.join(BASE_DIR, "data", "suppliers.csv")
inventory_path = os.path.join(BASE_DIR, "data", "inventory.csv")
sales_df = pd.read_csv(sales_path)
suppliers_df = pd.read_csv(suppliers_path)
inventory_df = pd.read_csv(inventory_path)

suppliers_df["supplier"] = suppliers_df["supplier"].str.title()

# -----------------------------
# PRODUCT SELECTOR
# -----------------------------
st.subheader("🛒 Select Product")
products = inventory_df["product"].tolist()
selected_product = st.selectbox(
    "Choose a product to optimize:",
    products,
    index=0
)

# Show selected product info
product_info = inventory_df[inventory_df["product"] == selected_product].iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Current Stock", f"{product_info['current_stock']} units")
col2.metric("⏱️ Lead Time", f"{product_info['lead_time']} days")
col3.metric("💰 Holding Cost", f"${product_info['holding_cost']}")
col4.metric("⚠️ Past Delays", f"{product_info['past_delay']}")

st.divider()

# -----------------------------
# RUN AI SYSTEM
# -----------------------------
if st.button("🚀 Run Agentic AI System"):

    with st.spinner(f"🤖 LLM Agent is optimizing supply chain for {selected_product}..."):
        state, messages = run_llm_agent(product=selected_product)

    demand = state["demand"]
    reorder = state["reorder"]
    supplier = state["supplier"]
    reliability = state["reliability"]

    # -----------------------------
    # METRICS DISPLAY
    # -----------------------------
    st.subheader("📊 Final Decision Metrics")
    st.caption(f"Product: **{selected_product}**")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🔮 Predicted Demand", f"{demand} units")
    col2.metric("📦 Reorder Quantity", f"{reorder} units")
    col3.metric("🏭 Selected Supplier", supplier.title() if supplier else "N/A")
    col4.metric("⭐ Supplier Reliability", f"{round(reliability, 2)}" if reliability else "N/A")

    if reliability and reliability < 0.6:
        st.warning("⚠️ Low reliability supplier detected — LLM re-evaluated and adjusted reorder quantity")
    elif reliability and reliability < 0.8:
        st.warning("⚠️ Medium reliability supplier — LLM applied safety buffer to reorder")
    else:
        st.success(f"✅ LLM Agent completed supply chain optimization for {selected_product} successfully")

    st.divider()

    # -----------------------------
    # LLM REASONING TRACE
    # -----------------------------
    st.subheader("🧠 LLM Agent Reasoning Trace")
    st.caption("Watch how the LLM agent thinks step by step and calls each tool autonomously")

    tool_results = []
    for msg in messages:
        if msg.get("role") == "tool":
            tool_results.append(msg.get("content"))

    step = 1
    result_index = 0

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        if role == "assistant" and tool_calls:
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = tc["function"]["arguments"]

                with st.expander(f"🔧 Step {step} — LLM calls: `{tool_name}`", expanded=False):
                    try:
                        args_parsed = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        if args_parsed:
                            st.json(args_parsed)
                        else:
                            st.caption("No arguments needed — tool runs automatically")
                    except:
                        st.caption("No arguments needed — tool runs automatically")

                if result_index < len(tool_results):
                    result = tool_results[result_index]
                    with st.expander(f"📦 Result from `{tool_name}`", expanded=False):
                        try:
                            st.json(json.loads(result))
                        except:
                            st.code(result)
                    result_index += 1

                step += 1

        elif role == "assistant" and content and not tool_calls:
            st.subheader("💡 LLM Final Recommendation")
            st.info(content)

    st.divider()

    # -----------------------------
    # DEMAND GRAPH
    # -----------------------------
    st.subheader("📈 Demand Forecast Visualization")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sales_df["sales"], marker='o', markersize=3,
            label="Historical Sales", color="steelblue", linewidth=1.5)
    if demand:
        ax.axhline(y=demand, color='red', linestyle='--',
                   linewidth=2, label=f"Predicted Demand: {demand} units")
    ax.set_title(f"Historical Sales vs Predicted Demand — {selected_product}", fontsize=14)
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Sales Units")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # -----------------------------
    # SUPPLIER COMPARISON
    # -----------------------------
    st.subheader("🏭 Supplier Comparison")

    suppliers_df = pd.read_csv(suppliers_path)
    suppliers_df["supplier"] = suppliers_df["supplier"].str.title()

    colors_cost = ["red" if s.lower() == supplier.lower() else "steelblue"
                   for s in suppliers_df["supplier"]]
    colors_rel = ["red" if s.lower() == supplier.lower() else "green"
                  for s in suppliers_df["supplier"]]

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(suppliers_df["supplier"], suppliers_df["cost"], color=colors_cost)
    axes[0].set_title("Supplier Cost Comparison", fontsize=13)
    axes[0].set_xlabel("Supplier")
    axes[0].set_ylabel("Cost")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(suppliers_df["supplier"], suppliers_df["reliability"], color=colors_rel)
    axes[1].set_title("Supplier Reliability Comparison", fontsize=13)
    axes[1].set_xlabel("Supplier")
    axes[1].set_ylabel("Reliability Score")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig2)

    st.caption("🔴 Red bar = Selected supplier by LLM")

    st.divider()

    # -----------------------------
    # SUPPLIER DATA TABLE
    # -----------------------------
    st.subheader("📋 Supplier Data")
    st.dataframe(suppliers_df, use_container_width=True)


# -----------------------------
# SHOW RAW SALES DATA
# -----------------------------
st.divider()
st.subheader("📂 Historical Sales Data")
st.dataframe(sales_df, use_container_width=True)