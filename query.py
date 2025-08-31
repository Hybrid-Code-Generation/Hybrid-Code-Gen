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
    print(f"{i}. {method_info['Class']}.{method_info['Method Name']} -> Context: {result}\n")

# Now we will get the information for each method returned by the KG search
from search_method import search_method as search_method_csv

detailed_results = []
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

    for inner_method_info in context.get('CALLS', []) + context.get('CALLED_BY', []):
        print(f"Fetching detailed info for related method: {inner_method_info['class_name']}.{inner_method_info['method_name']}.{inner_method_info['parameters']}.{inner_method_info['return_type']}")
        inner_detailed_info = search_method_csv(
            method_name=inner_method_info['method_name'],
            class_name=inner_method_info['class_name'],
            parameters=inner_method_info['parameters'],
            return_type=inner_method_info['return_type']
        )
        print(f"Detailed Info for {inner_method_info['class_name']}.{inner_method_info['method_name']}: {json.dumps(inner_detailed_info, indent=2)}\n")
        detailed_results.append({
            'method_info': inner_method_info,
            'context': {},
            'detailed_method_info': inner_detailed_info
        })

# Save the detailed results to a JSON file for further use
with open('detailed_search_results.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)
