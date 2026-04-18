from inference import run
import sys

print("=== Pocket-Agent Chatbot Demo ===")
print("Ask me about weather, currency, conversions, etc. (Type 'exit' to stop)")

history = []
while True:
    try:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        response = run(user_input, history)
        print(f"Assistant: {response}")
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
    except KeyboardInterrupt:
        break