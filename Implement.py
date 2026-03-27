import ast
import re
import spacy
#from rebel import RebelExtractor
import networkx as nx
from groq import Groq

# Graph is Global now for simplicity, but in production you'd want a more robust state management
G = nx.DiGraph()

def graph_query(entity):
    results = []
    
    for neighbor in G.neighbors(entity):
        relation = G[entity][neighbor]["relation"]
        results.append((entity, relation, neighbor))
        
    return results

def incoming(entity):
    results = []
    
    for src in G.predecessors(entity):
        relation = G[src][entity]["relation"]
        results.append((src, relation, entity))
        
    return results

def multi_hop_reasoning(graph,target,hops=2):

    paths=[]
    for node in graph.nodes:

        try:
            path=nx.shortest_path(graph,node,target)
            if len(path)-1 <= hops:
                paths.append(path)
        except:
            pass

    print(f"Multi hop reasoning paths: {paths}\n")
    return paths

def multi_hop_reasoning_all_targets(graph, targets, hops=4):
    all_paths = []

    # paths BETWEEN targets (e.g. A → C)
    for i in range(len(targets)):
        for j in range(len(targets)):
            if i != j:
                try:
                    path = nx.shortest_path(graph, targets[i], targets[j])
                    if len(path) - 1 <= hops:
                        all_paths.append(path)
                except:
                    pass

    # paths FROM each target to ALL other nodes in graph
    for target in targets:
        for node in graph.nodes:
            if node == target:
                continue
            try:
                path = nx.shortest_path(graph, target, node)
                if 1 < len(path) - 1 <= hops:
                    all_paths.append(path)
            except:
                pass

        # paths TO each target from all nodes
        for node in graph.nodes:
            if node == target:
                continue
            try:
                path = nx.shortest_path(graph, node, target)
                if 1 < len(path) - 1 <= hops:
                    all_paths.append(path)
            except:
                pass

    # deduplicate
    seen = set()
    unique = []
    for path in all_paths:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique

# def find_target(question, graph):
#     for node in graph.nodes:
#         if node.lower() in question.lower():
#             return node
#     return None

# def find_all_targets(question, graph):
#     question_lower = question.lower()
#     matches = []

#     for node in graph.nodes:
#         if node.lower() in question_lower:
#             matches.append(node)

#     # sort by length descending so "vscode users" is preferred over "vscode"
#     matches.sort(key=lambda x: len(x), reverse=True)

#     # remove nodes that are substrings of a longer match already in the list
#     filtered = []
#     for node in matches:
#         if not any(node.lower() in other.lower() and node != other for other in filtered):
#             filtered.append(node)

#     return filtered

def find_all_targets(question, nodes):
    system_prompt_target = f"""You are a system that selects relevant nodes from a knowledge graph.
        Graph Nodes:
        {nodes}
        Task:
        Identify which nodes from the graph are mentioned or referred to in the question that will be given to you.
        Rules:
        - Only return nodes that exist in the provided Graph Nodes list.
        - If multiple nodes match, return all of them.
        - If none match, return an empty list.
        - Do NOT invent or modify node names.

        Correct Output format:
        <OUTPUT>["node1","node2","node3"]</OUTPUT>
        Wrong Output formats (do NOT follow these):
        <OUTPUT>node1, node2, node3</OUTPUT>
        <OUTPUT>[node1, node2, node3]</OUTPUT>
        <OUTPUT>[("node1"), "node2", ("node3")]</OUTPUT>
        """
    
    groq_response = groq_call(system_prompt_target, [], question).choices[0].message.content
    raw = extract_llm_outputs(groq_response)

    if not raw:
        print("No <OUTPUT> found in target response, returning []")
        return []

    try:
        result = ast.literal_eval(raw)

        if not isinstance(result, list):
            print(f"Unexpected type from LLM: {type(result)}, returning []")
            return []

        # validate — only keep nodes that actually exist in the graph
        valid_nodes = set(nodes)
        filtered = [n for n in result if n in valid_nodes]

        if len(filtered) != len(result):
            hallucinated = [n for n in result if n not in valid_nodes]
            print(f"Hallucinated nodes removed: {hallucinated}")

        print(f"Validated targets: {filtered}")
        return filtered

    except Exception as e:
        print(f"Failed to parse target response: {e}\nRaw: {raw}")
        return []

def extract_llm_outputs(llm_output):
    match = re.search(r'<OUTPUT>(.*?)</OUTPUT>', llm_output, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def groq_call(system_prompt, context, chunk_content, tools=[], tool_choice="none"):
    client = Groq(
        api_key="GROQ_API_KEY"
    )

    message = [
            {"role": "system", 
             "content": system_prompt
             }
        ] + context 
    # + [
    #         {"role": "user", 
    #          "content": chunk_content
    #          }
    #     ]
    
    if chunk_content:
        message.append({
            "role": "user",
            "content": chunk_content
        })
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=message,
        tools=tools,
        tool_choice=tool_choice,
    )
    #return response.choices[0].message.content
    return response

def triplet_via_llm(chunk_content):

    system_prompt = """
        You are an information extraction system for a knowledge graph.
        Your task is to extract factual triplets from the input text.
        Rules:
        1. Each fact must be represented as a triplet:(subject, relation, object)
        2. Use clear entity names as subjects and objects.
        3. If a sentence contains multiple facts, extract multiple triplets.
        4. Resolve pronouns when possible.
            Example: "They made it free" → replace pronouns with actual entity names.
            IMPORTANT: When resolving pronouns like "it" or "they", the subject of the new triplet should be the most recently mentioned OBJECT, not the original subject.
        5. Relations must be short and joined with underscores.
            Example:is_a,created_by,free_for,launched_by
        6. Do NOT merge multiple concepts into one entity.
        7. Do Not modify the text with the actual information. The chunk provided for you is the actual information.
        8. Return ONLY the triplets.
        9. The triplets should be in the format: [(subject, relation, object)]. Very important: The output should be a list of triplets in the exact format as shown in the example below. Do not include any additional text or explanation, only the list of triplets.
        10. All the subject, relation, object should be in string format and should be wrapped in double quotes, not in “”. this is an ERROR (invalid character '“' (U+201C))
        Wrap the output inside:

        <OUTPUT>
        [(triplet1),
        (triplet2)]
        </OUTPUT>"""

    context = [{
        'role': 'user',
        'content': "Openai's GPT-oss-20b is a reasoning model",
    },
    {
        'role': 'assistant',
        'content': '<OUTPUT> [("OpenAI", "created", "GPT-oss-20b"),("GPT-oss-20b", "is_a", "reasoning model")] </OUTPUT>',
    }]

    
    groq_response = groq_call(system_prompt, context, chunk_content).choices[0].message.content

    print("LLM Output for Triplet Extraction:", groq_response)
    triplets = extract_llm_outputs(groq_response)
    try:
        return ast.literal_eval(triplets)
    except:
        fixed = re.sub(r'\((\s*)([A-Za-z0-9_ ]+)(\s*),', r'("\2",', triplets)
        return ast.literal_eval(fixed)

def answer_question(question,paths):
    print(f"Paths for reasoning: {paths}")

    context = ""
    for path in paths:
        if len(path) > 1:
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                relation = G[src][dst]["relation"]
                context += f"{src} --[{relation}]--> {dst}\n"

    prompt=f"""you are an question answering LLM. you have to answer the question only based on the context that is provided to u. Should not user your inbuilt knowledge to answer the question. If you don't know the answer based on the context, say you don't know.
Context:
{context}
Answer based on the graph context. Wrap your answer in <OUTPUT></OUTPUT> tags.
"""
    groq_response = groq_call(prompt, [], question).choices[0].message.content

    return groq_response

def upload_document_to_graph(document):
    try:
        triplet = triplet_via_llm(document)
        # better — purpose built relation extraction model
        # triplet = RebelExtractor().extract(document)
        print("Extracted Triplets:", triplet)

        for s, r, o in triplet:
            G.add_edge(s, o, relation=r)
        print("Current Graph Nodes after Upload:", G.nodes)
        print("UPLOAD COMPLETE\n")
        return {"status": "success", "nodes": list(G.nodes), "edges": len(G.edges)}  # ← JSON serializable
    except Exception as e:
        print(f"Error uploading document to graph: {e}")

def query_from_graph(query):
    target_node = find_all_targets(query, G.nodes)
    print("Identified Target Node in Graph:", target_node, "\n")
    paths = multi_hop_reasoning_all_targets(G, target_node, hops=5) #  hops - 3 means we are looking for paths of length 3 or less for deeper reasoning
    answer = answer_question(query, paths)
    print("\nAnswer to the Query:", answer)

    # strip <OUTPUT> tags before returning to agent
    clean = extract_llm_outputs(answer)
    return {"answer": clean if clean else answer}

# while True:
#     n = int(input("How many documents do you want to add? "))
#     for _ in range(n):
#         document = input("Enter the Document Content: ")
#         upload_document_to_graph(document)

#     user_query = input("Enter your Query: ")
#     target_node = find_all_targets(user_query, G.nodes)
#     print("Identified Target Node in Graph:", target_node, "\n")
#     # this is for providing the context for the LLM to infer and answer the query based on it.
#     paths = multi_hop_reasoning_all_targets(G, target_node, hops=5) #  hops - 3 means we are looking for paths of length 3 or less for deeper reasoning
#     answer = answer_question(user_query, paths)
#     print("\nAnswer to the Query:", answer)
