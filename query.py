# This file is responsible for getting a user query as input and then generating embeddings for it using Azure OpenAI and then compare with existing embeddings and return top K similar method names. After this, we'll use these method names to get methods which are being called by these methods and its parent methods etc. to create a prompt for code generation.

# We have a ready made function availble for generating embeddings and getting top K similar methods.

import json
from query_search_OpenAI import CodeSearcher

searcher = CodeSearcher()
searcher.initialize()
# Clear any previously discovered types to start fresh
searcher.clear_discovered_types()
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
all_methods_for_type_extraction = []  # Collect all methods for type extraction

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
    
    # Add this method to our collection for type extraction
    if detailed_method_info:
        all_methods_for_type_extraction.append(detailed_method_info)

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
        detailed_results.append({
            'method_info': inner_method_info,
            'context': {},
            'detailed_method_info': inner_detailed_info,
            'similarity_score': similarity
        })
        
        # Add this related method to our collection for type extraction
        if inner_detailed_info:
            all_methods_for_type_extraction.append(inner_detailed_info)

# Extract non-primitive types from ALL discovered methods AFTER KG traversal
        print(f"Similarity score: {similarity}\n")
        # print(f"Java Method String: {}\n")
        inner_method_str = convert_json_to_java_method_str(inner_detailed_info)
        final_prompt_to_llm += f"\n{inner_method_str}\n"

final_prompt_to_llm = f"The following are Java methods relevant to the user's query: '{user_query}'. \n\nUse these methods to assist in code generation.\n\n\n{final_prompt_to_llm}"
unique_non_primitive_types = searcher.extract_types_from_all_methods(all_methods_for_type_extraction)

# Store the unique types in the searcher's memory for later access
print(f"\n💾 Storing {sum(len(v) for v in unique_non_primitive_types.values())} unique types in searcher memory...")

# Get specific categories for analysis
print(f"\n📚 Classes discovered: {len(unique_non_primitive_types['classes'])} - {unique_non_primitive_types['classes'][:10] if len(unique_non_primitive_types['classes']) > 10 else unique_non_primitive_types['classes']}")
print(f"🗂️ Collections discovered: {len(unique_non_primitive_types['collections'])} - {unique_non_primitive_types['collections']}")
print(f"🏷️ Annotations discovered: {len(unique_non_primitive_types['annotations'])} - {unique_non_primitive_types['annotations']}")
        

# Save the detailed results to a JSON file for further use
with open('detailed_search_results.json', 'w') as f:
    # Add the unique types summary to the results
    final_results = {
        'query': user_query,
        'unique_non_primitive_types': unique_non_primitive_types,
        'total_methods_analyzed': len(all_methods_for_type_extraction),
        'detailed_results': detailed_results
    }
    json.dump(final_results, f, indent=2)

print(f"\n💾 Results saved to 'detailed_search_results.json' with {len(all_methods_for_type_extraction)} methods analyzed")
print(f"🧠 Non-primitive types are now stored in searcher memory and can be accessed via:")
print(f"   - searcher.get_discovered_types()")
print(f"   - searcher.get_discovered_types('classes')")
print(f"   - Or from the JSON file: detailed_search_results.json")

with open('prompt_to_llm.txt', 'w') as f:
    f.write(final_prompt_to_llm)
