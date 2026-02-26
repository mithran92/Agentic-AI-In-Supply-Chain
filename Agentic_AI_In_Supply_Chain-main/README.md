# Agentic AI in Supply Chain

An autonomous LLM-driven supply chain optimization system that uses AI agents to predict demand, manage inventory, select suppliers, and make decisions autonomously.

## Architecture
```
LLM Agent (Groq - llama3.3-70b)
├── Demand Agent — LSTM model for demand forecasting
├── Inventory Agent — Random Forest for reorder quantity
├── Supplier Agent — Random Forest for supplier selection
└── Feedback Agent — Reliability scoring and updates
```

The LLM acts as the brain — it autonomously calls each agent as a tool, reasons about the results, and makes the final supply chain decision.

## Features

- LLM-driven autonomous decision making (Level 3 Agentic AI)
- LSTM demand prediction from historical sales data
- Random Forest inventory optimization
- Supplier selection based on cost, reliability and reorder quantity
- Feedback loop updating supplier reliability over time
- Real-time reasoning trace visible in dashboard
- Streamlit dashboard with charts and metrics

## Setup

### 1. Clone the repository
```
git clone <your-repo-url>
cd Agentic_AI_In_Supply_Chain-main
```

### 2. Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key at: https://console.groq.com

### 5. Run the dashboard
```
streamlit run dashboard/app.py
```

### 6. Or run the terminal version
```
python main.py
```

## Project Structure