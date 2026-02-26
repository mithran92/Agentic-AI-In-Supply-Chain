import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Import all agents as tools
from agents.advanced_demand_agent import predict_demand_lstm
from agents.inventory_agent import inventory_decision
from agents.supplier_agent import select_supplier
from agents.feedback_agent import update_reliability

# Import memory
from llm.memory import load_memory, save_memory, format_memory_for_llm

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# TOOL DEFINITIONS — what LLM can call
# ─────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "predict_demand",
            "description": "Predicts future product demand using LSTM model based on historical sales data",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_reorder",
            "description": "Calculates the reorder quantity based on predicted demand and current inventory levels",
            "parameters": {
                "type": "object",
                "properties": {
                    "predicted_demand": {
                        "type": "integer",
                        "description": "The predicted demand value from the demand agent"
                    }
                },
                "required": ["predicted_demand"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_best_supplier",
            "description": "Selects the best supplier based on cost, delivery time, reliability and reorder quantity",
            "parameters": {
                "type": "object",
                "properties": {
                    "reorder_qty": {
                        "type": "integer",
                        "description": "The reorder quantity to help select the most suitable supplier"
                    }
                },
                "required": ["reorder_qty"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_supplier_reliability",
            "description": "Updates and returns the reliability score of the selected supplier based on past performance",
            "parameters": {
                "type": "object",
                "properties": {
                    "supplier_name": {
                        "type": "string",
                        "description": "The name of the selected supplier"
                    }
                },
                "required": ["supplier_name"]
            }
        }
    }
]


# ─────────────────────────────────────────
# TOOL EXECUTION — runs the actual agent
# ─────────────────────────────────────────

def execute_tool(tool_name, tool_args):

    if tool_name == "predict_demand":
        result = predict_demand_lstm()
        return {"demand": result}

    elif tool_name == "calculate_reorder":
        demand = tool_args.get("predicted_demand")
        decision, reorder = inventory_decision(demand)
        return {"decision": decision, "reorder_qty": reorder}

    elif tool_name == "select_best_supplier":
        reorder = tool_args.get("reorder_qty")
        supplier, reliability, suppliers_df = select_supplier(reorder)
        return {"supplier": supplier, "reliability": reliability}

    elif tool_name == "update_supplier_reliability":
        supplier_name = tool_args.get("supplier_name")
        updated_reliability = update_reliability(supplier_name)
        return {"updated_reliability": updated_reliability}

    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ─────────────────────────────────────────
# MAIN LLM AGENT LOOP — Level 3 + Memory
# ─────────────────────────────────────────

def run_llm_agent():

    print("\n=== LLM Agent Starting ===\n")

    # Load memory from previous runs
    memory = load_memory()
    memory_text = format_memory_for_llm(memory)
    print(f"Memory loaded: {len(memory)} previous runs found")

    messages = [
        {
            "role": "system",
            "content": f"""You are an autonomous supply chain AI agent.

Your job is to manage the supply chain by using the tools available to you.

You have access to memory from previous runs:
{memory_text}

Use this memory to:
- Compare current demand with previous demand trends
- Notice if the same supplier keeps getting selected
- Flag if reliability is dropping over time
- Make smarter decisions based on patterns you observe

Follow this process:
1. First predict demand using the predict_demand tool
2. Then calculate reorder quantity using calculate_reorder tool
3. Then select the best supplier using select_best_supplier tool
4. Then update supplier reliability using update_supplier_reliability tool
5. If supplier reliability is below 0.6, call calculate_reorder and select_best_supplier again
6. Finally provide a clear professional summary of all decisions made, compare with previous runs if available, and explain any trends you notice

You must call the tools yourself. Think step by step. Be autonomous.
Do NOT pass supplier_reliability to calculate_reorder."""
        },
        {
            "role": "user",
            "content": "Please run the full supply chain optimization process and give me your final recommendation."
        }
    ]

    state = {
        "demand": None,
        "reorder": None,
        "supplier": None,
        "reliability": None,
        "reasoning": []
    }

    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:

        iteration += 1
        print(f"--- LLM Thinking (iteration {iteration}) ---")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1000
        )

        message = response.choices[0].message

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in (message.tool_calls or [])
            ] or None
        })

        if not message.tool_calls:
            print("\n=== LLM Final Decision ===")
            print(message.content)
            state["reasoning"].append(message.content)
            break

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"LLM calling tool: {tool_name} with args: {tool_args}")

            tool_result = execute_tool(tool_name, tool_args)

            print(f"Tool result: {tool_result}")

            if "demand" in tool_result:
                state["demand"] = tool_result["demand"]
            if "reorder_qty" in tool_result:
                state["reorder"] = tool_result["reorder_qty"]
            if "supplier" in tool_result:
                state["supplier"] = tool_result["supplier"]
            if "reliability" in tool_result:
                state["reliability"] = tool_result["reliability"]
            if "updated_reliability" in tool_result:
                state["reliability"] = tool_result["updated_reliability"]

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result)
            })

    # Save this run to memory
    if state["demand"] and state["reorder"] and state["supplier"]:
        save_memory(
            state["demand"],
            state["reorder"],
            state["supplier"],
            state["reliability"]
        )
        print("Memory saved for this run")

    return state, messages


# ─────────────────────────────────────────
# SIMPLE WRAPPER — for dashboard/app.py
# ─────────────────────────────────────────

def ask_llm(demand=None, reorder=None, supplier=None):
    state, messages = run_llm_agent()

    final_explanation = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_explanation = msg["content"]
            break

    return final_explanation, state