from llm.llm_helper import run_llm_agent

print("=== Agentic AI Supply Chain ===")
print("LLM Agent is now in control...\n")

state, messages = run_llm_agent()

print("\n=== Final State ===")
print("Demand:", state["demand"])
print("Reorder:", state["reorder"])
print("Supplier:", state["supplier"])
print("Reliability:", state["reliability"])