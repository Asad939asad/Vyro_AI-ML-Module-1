import os
from llama_cpp import Llama

# 1. INITIALIZATION: Load the model
MODEL_PATH = "/content/pocket_agent_q4.gguf"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=4,
    verbose=False
)

# 2. FINAL SYSTEM PROMPT: Complete few-shot grounding for all 5 tools
SYSTEM_PROMPT = """You are a strict tool-calling agent. Output ONLY valid JSON wrapped in <tool_call>...</tool_call> tags.

Tool Schemas:
- weather: {"tool": "weather", "args": {"location": "string", "unit": "C|F"}}
- calendar: {"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string"}}
- convert: {"tool": "convert", "args": {"value": number, "from_unit": "string", "to_unit": "string"}}
- currency: {"tool": "currency", "args": {"amount": number, "from": "ISO3", "to": "ISO3"}}
- sql: {"tool": "sql", "args": {"query": "string"}}

Examples:
User: What is the weather in London in Celsius?
Assistant: <tool_call>{"tool": "weather", "args": {"location": "London", "unit": "C"}}</tool_call>

User: Schedule a meeting for 2026-06-01 called Strategy Session.
Assistant: <tool_call>{"tool": "calendar", "args": {"action": "create", "date": "2026-06-01", "title": "Strategy Session"}}</tool_call>

User: Convert 5 kilometers to miles.
Assistant: <tool_call>{"tool": "convert", "args": {"value": 5, "from_unit": "kilometers", "to_unit": "miles"}}</tool_call>

User: How much is 100 USD in EUR?
Assistant: <tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}}</tool_call>

User: Show me all employees in the sales department.
Assistant: <tool_call>{"tool": "sql", "args": {"query": "SELECT * FROM employees WHERE department = 'sales';"}}</tool_call>

User: Tell me a joke.
Assistant: I cannot engage in casual conversation.

User: Who is the president of the US?
Assistant: I cannot answer general knowledge or factual questions.

Rules:
1. NEVER use 'currency' for physical distances like miles or km.
2. ONLY use 'sql' if the user explicitly asks to query a database, table, or system records. DO NOT use 'sql' for trivia or general knowledge.
3. For 'convert', use full unit names as mentioned (e.g., "miles", "kilometers").
4. Output ONLY the tool call or a brief refusal. No conversational filler."""

# 3. CORE RUN FUNCTION
def run(prompt: str, history: list[dict]) -> str:
    # Build ChatML prompt
    full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    
    for entry in history:
        full_prompt += f"<|im_start|>{entry['role']}\n{entry['content']}<|im_end|>\n"
        
    full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Deterministic generation
    output = llm(
        full_prompt,
        max_tokens=150,
        temperature=0.0,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    return output["choices"][0]["text"].strip()

# 4. BATCH TEST (For your final verification)
if __name__ == "__main__":
    test_prompts = [
        "How many kilometers are in 25 miles?",
        "What's the weather like in Lahore?",
        "Convert 500 PKR to USD",
        "Add an event for 2026-12-25 called Christmas Dinner",
        "Select all users from the orders table",
        "Who is the president of the US?"
    ]
    
    print("--- FINAL SYSTEM TEST ---")
    for tp in test_prompts:
        print(f"User: {tp}")
        print(f"Assistant: {run(tp, [])}\n")