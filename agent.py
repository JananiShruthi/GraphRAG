import json

from tools import scrape_webpage, calculator, search_web, get_current_time
from groq import Groq
from Implement import query_from_graph, upload_document_to_graph, groq_call

# ─── Tool Definitions for Groq ──────────────────────────────────────────────────────────
tools = [{
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression like '2 + 2' or '10 * 5 / 2'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "upload_document_to_graph",
        "description": "Extract triplets from a document and add them to the knowledge graph. Call this after getting document content — whether from raw text, a scraped URL, or a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "document": {
                    "type": "string",
                    "description": "The raw text content to extract triplets from and add to the graph"
                }
            },
            "required": ["document"]
        }
    }},
    {
    "type": "function",
    "function": {
        "name": "query_from_graph",
        "description": "Query the knowledge graph to answer a question using multi-hop reasoning. Call this when the user asks a question about the document that has already been uploaded to the graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question to answer from the knowledge graph"
                }
            },
            "required": ["query"]
        }
    }}]

# ─── Tool Router ───────────────────────────────────────────────────────────────

tool_map = {
    "get_current_time": get_current_time,
    "calculator": calculator,
    "upload_document_to_graph": upload_document_to_graph,
    "query_from_graph": query_from_graph,
}

system_prompt_agent = """You are a GraphRAG agent with access to a knowledge graph and tools.
IMPORTANT:
- You can call ONLY ONE tool at a time.
- After calling a tool, wait for the result before deciding the next step.
- Do NOT call multiple tools in a single response.
- DO NOT simulate or write function calls as text.
STRICT RULES:
1. User uploads a document → call upload_document_to_graph
2. User provides a URL → call scrape_webpage, then upload_document_to_graph
3. User asks about time/date → call get_current_time (NOT query_from_graph)
4. User asks a calculation → call calculator (NOT query_from_graph)
5. User asks about people, places, facts from uploaded documents → call query_from_graph
6. User asks for real-time info not in graph → call search_web
7. NEVER call query_from_graph for time, math, or web searches

EXAMPLES:
"what time is it?" → get_current_time
"what is 2+2?" → calculator
"who is person?" → query_from_graph
"upload this doc" → upload_document_to_graph
"upload doc and tell time" → upload_document_to_graph AND get_current_time"""

conversation_history = []

while True:
    user_input = input("Enter your query (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    conversation_history.append({"role": "user", "content": user_input})
    i = 0
    while True:
        response = groq_call(system_prompt_agent, conversation_history, "", tools=tools, tool_choice="auto")
        message = response.choices[0].message

        # 🧠 store assistant decision
        if message.tool_calls:
            conversation_history.append({
                "role": "assistant",
                "tool_calls": message.tool_calls
            })

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) or {}

                print(f"🔧 Calling: {name}({args})")
                result = tool_map[name](**args)
                print(f"   Result: {result}\n")

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            continue  # 🔥 keep reasoning

        # ⚠️ recover from fake tool JSON
        if message.content and '"expression"' in message.content:
            print("⚠️ Retrying due to fake tool output...")
            continue

        # ✅ final answer
        print(f"\n🤖 Agent: {message.content}\n")
        conversation_history.append({
            "role": "assistant",
            "content": message.content
        })
        break
