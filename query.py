# This file is responsible for getting a user query as input and then generating embeddings for it using Azure OpenAI and then compare with existing embeddings and return top K similar method names. After this, we'll use these method names to get methods which are being called by these methods and its parent methods etc. to create a prompt for code generation.

# We have a ready made function availble for generating embeddings and getting top K similar methods.

import json
import pandas as pd
import re
from query_search_OpenAI import CodeSearcher

searcher = CodeSearcher()
searcher.initialize()
K = 3

# Load class names from class.csv for exact matching
def load_class_names():
    """Load class names from class.csv file"""
    try:
        class_df = pd.read_csv('class.csv')
        # Extract unique class names from the 'Class' column
        class_names = set(class_df['Class'].dropna().unique())
        print(f"✅ Loaded {len(class_names)} unique class names from class.csv")
        return class_names
    except Exception as e:
        print(f"❌ Error loading class names: {e}")
        return set()

def find_exact_class_matches(method_info, class_names):
    """
    Find exact string matches of class names in method body, parameters, and return type
    Args:
        method_info: Dictionary containing method details
        class_names: Set of class names to search for
    Returns:
        Set of matched class names
    """
    matched_classes = set()
    
    if not method_info or not class_names:
        return matched_classes
    
    # Get method components to search in
    return_type = method_info.get('Return Type', '') or ''
    parameters = method_info.get('Parameters', '') or ''
    function_body = method_info.get('Function Body', '') or ''
    
    # Convert to strings in case of NaN values
    return_type = str(return_type) if return_type else ''
    parameters = str(parameters) if parameters else ''
    function_body = str(function_body) if function_body else ''
    
    # Combine all text to search in
    search_text = f"{return_type} {parameters} {function_body}"
    
    # Find exact matches for each class name
    for class_name in class_names:
        if not class_name or class_name.lower() == 'nan':
            continue
            
        # Use word boundary regex to find exact matches
        # \b ensures we match whole words only (e.g., "User" but not "GetUser")
        pattern = r'\b' + re.escape(class_name) + r'\b'
        
        if re.search(pattern, search_text, re.IGNORECASE):
            matched_classes.add(class_name)
    
    return matched_classes

def get_class_bodies_for_matched_classes(matched_classes):
    """
    Get class bodies from class.csv for the matched class names
    Args:
        matched_classes: Set of matched class names
    Returns:
        Dictionary mapping class names to their class bodies
    """
    class_bodies = {}
    
    if not matched_classes:
        return class_bodies
    
    try:
        class_df = pd.read_csv('class.csv')
        
        # Filter for the matched classes
        for class_name in matched_classes:
            class_row = class_df[class_df['Class'] == class_name]
            if not class_row.empty:
                class_body = class_row.iloc[0]['ClassBody']
                class_bodies[class_name] = class_body
                print(f"📋 Found class body for: {class_name}")
            else:
                print(f"⚠️ Class body not found for: {class_name}")
        
    except Exception as e:
        print(f"❌ Error loading class bodies: {e}")
    
    return class_bodies

def convert_class_to_java_str(class_name, class_body):
    """
    Convert class information to Java class string format
    Args:
        class_name: Name of the class
        class_body: Body of the class
    Returns:
        Formatted Java class string
    """
    if not class_body or str(class_body).lower() == 'nan':
        return f"class {class_name} {{ /* Class body not available */ }}"
    
    return f"class {class_name} {class_body}"

# Load class names once at the beginning
available_class_names = load_class_names()
unique_matched_classes = set()  # Global set to store all unique matched classes

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
    
    # Find exact class name matches in this method
    if detailed_method_info and available_class_names:
        matched_classes = find_exact_class_matches(detailed_method_info, available_class_names)
        if matched_classes:
            unique_matched_classes.update(matched_classes)
            print(f"🎯 Found class matches in {method_info['Class']}.{method_info['Method Name']}: {matched_classes}")
    
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
        if similarity is None or (hasattr(similarity, 'item') and similarity.item() < 0.2) or (not hasattr(similarity, 'item') and similarity < 0.2):
             print(f"Skipping due to low similarity with name {inner_method_info['method_name']}\n")
             continue
        
        detailed_results.append({
            'method_info': inner_method_info,
            'context': {},
            'detailed_method_info': inner_detailed_info,
            'similarity_score': similarity
        })
        
        # Find exact class name matches in this related method
        if inner_detailed_info and available_class_names:
            matched_classes = find_exact_class_matches(inner_detailed_info, available_class_names)
            if matched_classes:
                unique_matched_classes.update(matched_classes)
                print(f"🎯 Found class matches in related method {inner_method_info['class_name']}.{inner_method_info['method_name']}: {matched_classes}")
        
        # print(f"Java Method String: {}\n")
        inner_method_str = convert_json_to_java_method_str(inner_detailed_info)
        final_prompt_to_llm += f"\n{inner_method_str}\n"

# Print summary of matched classes
print(f"\n📚 UNIQUE CLASS MATCHES FOUND:")
if unique_matched_classes:
    print(f"   Total unique classes found: {len(unique_matched_classes)}")
    for class_name in sorted(unique_matched_classes):
        print(f"   - {class_name}")
else:
    print("   No class matches found in any method")

# Get class bodies for all matched classes
print(f"\n� Extracting class bodies for matched classes...")
matched_class_bodies = get_class_bodies_for_matched_classes(unique_matched_classes)

# Add class information to the final prompt
class_info_section = ""
if matched_class_bodies:
    class_info_section = "\n\nRelevant Java Classes referenced in the methods:\n\n"
    for class_name in sorted(matched_class_bodies.keys()):
        class_str = convert_class_to_java_str(class_name, matched_class_bodies[class_name])
        class_info_section += f"{class_str}\n\n"
    print(f"✅ Added {len(matched_class_bodies)} class definitions to prompt")
else:
    print("ℹ️ No class bodies found to add to prompt")

# Complete the final prompt construction with class information
final_prompt_to_llm = f"The following are Java methods relevant to the user's query: '{user_query}'. \n\nUse these methods to assist in code generation.\n\n\n{final_prompt_to_llm}{class_info_section}"
        

# Save the detailed results to a JSON file for further use
with open('detailed_search_results.json', 'w') as f:
    final_results = {
        'query': user_query,
        'unique_matched_classes': sorted(list(unique_matched_classes)),
        'total_matched_classes': len(unique_matched_classes),
        'matched_class_bodies': {k: v for k, v in matched_class_bodies.items()},
        'total_class_bodies_found': len(matched_class_bodies),
        'detailed_results': detailed_results
    }
    json.dump(final_results, f, indent=2)

print(f"\n💾 Results saved to 'detailed_search_results.json'")
print(f"🧠 Final prompt saved to 'prompt_to_llm.txt'")
print(f"📚 {len(unique_matched_classes)} unique class matches saved to JSON")
print(f"📋 {len(matched_class_bodies)} class bodies extracted and saved")

with open('prompt_to_llm.txt', 'w') as f:
    f.write(final_prompt_to_llm)
