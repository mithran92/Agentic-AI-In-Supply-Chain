import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_PATH = os.path.join(BASE_DIR, "data", "memory.json")


def load_memory():
    """Load past decisions from memory file"""
    if not os.path.exists(MEMORY_PATH):
        return []

    try:
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    except:
        return []


def save_memory(demand, reorder, supplier, reliability):
    """Save current decision to memory file"""
    memory = load_memory()

    # Add new entry
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "demand": demand,
        "reorder": reorder,
        "supplier": supplier,
        "reliability": round(reliability, 2) if reliability else None
    }

    memory.append(entry)

    # Keep only last 10 runs
    memory = memory[-10:]

    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)


def format_memory_for_llm(memory):
    """Format memory into readable text for LLM prompt"""
    if not memory:
        return "No previous decisions available. This is the first run."

    lines = ["Previous supply chain decisions (most recent last):"]
    for i, entry in enumerate(memory[-5:], 1):
        lines.append(
            f"Run {i} ({entry['timestamp']}): "
            f"Demand={entry['demand']}, "
            f"Reorder={entry['reorder']}, "
            f"Supplier={entry['supplier']}, "
            f"Reliability={entry['reliability']}"
        )

    return "\n".join(lines)