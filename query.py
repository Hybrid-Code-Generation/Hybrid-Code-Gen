# This file is responsible for getting a user query as input and then generating embeddings for it using Azure OpenAI and then compare with existing embeddings and return top K similar method names. After this, we'll use these method names to get methods which are being called by these methods and its parent methods etc. to create a prompt for code generation.

# We have a ready made function availble for generating embeddings and getting top K similar methods.

import json
from query_search_OpenAI import CodeSearcher

searcher = CodeSearcher()
searcher.initialize()
K = 3

user_query = input("Enter your code search query: ")

# Vector embedding search
top_matches = searcher.search_top_k(user_query, k=K)
print(f"\nTop {K} matching methods:")

# Now we need to get the methods which are being called by these methods and its parent methods etc. to create a prompt for code generation.
from search_neo4j import search_method

results = []
for i, method_info in enumerate(top_matches, 1):
    # KG search
    print(f"Current Method: {method_info['Class']}.{method_info['Method Name']}")
    result = search_method(
        method_name=method_info['Method Name'],
        class_name=method_info['Class'],
        parameters=method_info['Parameters'],
        return_type=method_info['Return Type'],
        calls=3,
        called_by=1,
        belongs_to=True,
        uses=True
    )
    results.append({
        'method_info': method_info,
        'context': result
    })
    # print(f"{i}. {method_info['Class']}.{method_info['Method Name']} -> Context: {result}\n")

# Now we will get the information for each method returned by the KG search
from search_method import search_method as search_method_csv
from search_method import search_method_csv_weighted  # Import the missing function

final_prompt_to_llm = ""

def convert_json_to_java_method_str(m):
            if not m:
                return ""
            body = m.get('Function Body', '')
            # Check if body is NaN (float) or empty
            if isinstance(body, float) or not body or str(body).lower() == 'nan':
                return f"{m['Return Type']} {m['Method Name']}({m['Parameters']})"
            return f"{m['Return Type']} {m['Method Name']}({m['Parameters']}) {body}"

detailed_results = []

i = 1
for item in results:
    method_info = item['method_info']
    context = item['context']

    # Search back in AST extracted CSV to get full method details
    detailed_method_info = search_method_csv(
        method_name=method_info['Method Name'],
        class_name=method_info['Class'],
        parameters=method_info['Parameters'],
        return_type=method_info['Return Type']
    )
    detailed_results.append({
        'method_info': method_info,
        'context': context,
        'detailed_method_info': detailed_method_info
    })

    method_str = convert_json_to_java_method_str(detailed_method_info)
    # print(f"Java Method String: {method_str}\n")
    final_prompt_to_llm += f"This is the {i} relevant method for the user query.\n{method_str}\n\nAnd these are the methods it calls or is called by:\n"
    i += 1

    for inner_method_info in context.get('CALLS', []) + context.get('CALLED_BY', []):
        inner_detailed_info = search_method_csv_weighted(
            user_query,
            method_name=inner_method_info['method_name'],
            class_name=inner_method_info['class_name'],
            parameters=inner_method_info['parameters'],
            return_type=inner_method_info['return_type']
        )
        if not inner_detailed_info:
            continue
        similarity = inner_detailed_info.get('similarity_score', None)
        print(f"Similarity score: {similarity}\n")

        ## This is important line to filter out low similarity methods
        if not (similarity is not None and similarity >= 0.2):
             print(f"Skipping due to low similarity with name {inner_method_info['method_name']}\n")
             continue
        
        detailed_results.append({
            'method_info': inner_method_info,
            'context': {},
            'detailed_method_info': inner_detailed_info,
            'similarity_score': similarity
        })
        
        # print(f"Java Method String: {}\n")
        inner_method_str = convert_json_to_java_method_str(inner_detailed_info)
        final_prompt_to_llm += f"\n{inner_method_str}\n"

# Complete the final prompt construction
final_prompt_to_llm = f"The following are Java methods relevant to the user's query: '{user_query}'. \n\nUse these methods to assist in code generation.\n\n\n{final_prompt_to_llm}"
        

# Save the detailed results to a JSON file for further use
with open('detailed_search_results.json', 'w') as f:
    final_results = {
        'query': user_query,
        'detailed_results': detailed_results
    }
    json.dump(final_results, f, indent=2)

print(f"\n💾 Results saved to 'detailed_search_results.json'")
print(f"🧠 Final prompt saved to 'prompt_to_llm.txt'")

with open('prompt_to_llm.txt', 'w') as f:
    f.write(final_prompt_to_llm)
