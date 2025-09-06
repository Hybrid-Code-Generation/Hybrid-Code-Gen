# This file is responsible for getting a user query as input and then generating embeddings for it using Azure OpenAI and then compare with existing embeddings and return top K similar method names. After this, we'll use these method names to get methods which are being called by these methods and its parent methods etc. to create a prompt for code generation.

# We have a ready made function availble for generating embeddings and getting top K similar methods.

import pandas as pd
import re
from processor.query_search_OpenAI import CodeSearcher

searcher = CodeSearcher()
searcher.initialize()
# K = 3
sim = 0.2

# delete all .txt files in data folder
import os
for file in os.listdir('./data/'):
    if file.endswith('.txt'):
        os.remove(os.path.join('./data/', file))

# delete all .md files in data/llm_responses folder
if os.path.exists('./data/llm_responses'):
    for file in os.listdir('./data/llm_responses'):
        if file.endswith('.md'):
            os.remove(os.path.join('./data/llm_responses', file))

# Load class names from class.csv for exact matching
def load_class_names():
    """Load class names from class.csv file"""
    try:
        class_df = pd.read_csv('./data/class.csv')
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
        class_df = pd.read_csv('./data/class.csv')
        
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

# Pre-fetch all data with maximum K value (3)
print("🔍 Pre-fetching all data with K=3...")
MAX_K = 3
top_matches = searcher.search_top_k(user_query, k=MAX_K)

# Import required functions
from processor.search_neo4j import search_method
from processor.search_method import search_method as search_method_csv
from processor.search_method import search_method_csv_weighted

def convert_json_to_java_method_str(m):
    if not m:
        return ""
    body = m.get('Function Body', '')
    # Check if body is NaN (float) or empty
    if isinstance(body, float) or not body or str(body).lower() == 'nan':
        return f"{m['Return Type']} {m['Method Name']}({m['Parameters']})"
    return f"{m['Return Type']} {m['Method Name']}({m['Parameters']}) {body}"

# Pre-fetch all KG search results and detailed method information
print("📊 Fetching KG context and detailed method information...")
all_results = []
all_related_methods = []  # Store all related methods with their similarities

for i, method_info in enumerate(top_matches, 1):
    print(f"Processing method {i}/{MAX_K}: {method_info['Class']}.{method_info['Method Name']}")
    
    # KG search
    result = search_method(
        method_name=method_info['Method Name'],
        class_name=method_info['Class'],
        parameters=method_info['Parameters'],
        return_type=method_info['Return Type'],
        calls=MAX_K-i+1,
        called_by=1,
        belongs_to=True,
        uses=True
    )
    
    # Get detailed method info
    detailed_method_info = search_method_csv(
        method_name=method_info['Method Name'],
        class_name=method_info['Class'],
        parameters=method_info['Parameters'],
        return_type=method_info['Return Type']
    )
    
    all_results.append({
        'method_info': method_info,
        'context': result,
        'detailed_method_info': detailed_method_info,
        'index': i  # Store original index for priority
    })
    
    # Process related methods and get their detailed info with similarity scores
    for inner_method_info in result.get('CALLS', []) + result.get('CALLED_BY', []):
        inner_detailed_info = search_method_csv_weighted(
            user_query,
            method_name=inner_method_info['method_name'],
            class_name=inner_method_info['class_name'],
            parameters=inner_method_info['parameters'],
            return_type=inner_method_info['return_type']
        )
        if inner_detailed_info:
            similarity = inner_detailed_info.get('similarity_score', 0)
            all_related_methods.append({
                'method_info': inner_method_info,
                'detailed_method_info': inner_detailed_info,
                'similarity_score': similarity,
                'parent_index': i  # Which main method this is related to
            })

print(f"✅ Pre-fetched {len(all_results)} main methods and {len(all_related_methods)} related methods")

# Now iterate through different K and sim values using the pre-fetched data
for K in range(1, 4, 1):
    for sim in [0.1, 0.2, 0.3, 0.4]:
        print(f"\n{'='*60}")
        print(f"Processing K={K}, similarity threshold={sim}")
        print(f"{'='*60}")
        
        # Filter main methods based on current K
        current_main_methods = [result for result in all_results if result['index'] <= K]
        
        # Filter related methods based on similarity threshold and parent method inclusion
        current_related_methods = [
            method for method in all_related_methods 
            if (method['similarity_score'] is not None and 
                ((hasattr(method['similarity_score'], 'item') and method['similarity_score'].item() >= sim) or 
                 (not hasattr(method['similarity_score'], 'item') and method['similarity_score'] >= sim)) and
                method['parent_index'] <= K)
        ]
        
        print(f"📊 Using {len(current_main_methods)} main methods and {len(current_related_methods)} related methods")
        
        # Build the prompt for current K and sim values
        final_prompt_to_llm = ""
        unique_matched_classes_per_run = set()
        
        # Process main methods
        for i, item in enumerate(current_main_methods, 1):
            method_info = item['method_info']
            detailed_method_info = item['detailed_method_info']
            
            # Find exact class name matches in this method
            if detailed_method_info and available_class_names:
                matched_classes = find_exact_class_matches(detailed_method_info, available_class_names)
                if matched_classes:
                    unique_matched_classes.update(matched_classes)
                    unique_matched_classes_per_run.update(matched_classes)
                    print(f"🎯 Found class matches in {method_info['Class']}.{method_info['Method Name']}: {matched_classes}")
            
            method_str = convert_json_to_java_method_str(detailed_method_info)
            final_prompt_to_llm += f"This is the {i} relevant method for the user query.\n\nFile name: {detailed_method_info['FilePath']}\n\nMethod: {method_str}\n\nAnd these are the methods it calls or is called by:\n"
        
        # Process related methods
        for method in current_related_methods:
            detailed_method_info = method['detailed_method_info']
            similarity = method['similarity_score']
            
            print(f"Including related method with similarity score: {similarity}")
            
            # Find exact class name matches in this related method
            if detailed_method_info and available_class_names:
                matched_classes = find_exact_class_matches(detailed_method_info, available_class_names)
                if matched_classes:
                    unique_matched_classes.update(matched_classes)
                    unique_matched_classes_per_run.update(matched_classes)
                    print(f"🎯 Found class matches in related method {method['method_info']['class_name']}.{method['method_info']['method_name']}: {matched_classes}")
            
            inner_method_str = convert_json_to_java_method_str(detailed_method_info)
            final_prompt_to_llm += f"File name: {detailed_method_info['FilePath']}\nMethod: {inner_method_str}\n"

        # Print summary of matched classes for this run
        print(f"\n📚 UNIQUE CLASS MATCHES FOUND (K={K}, sim={sim}):")
        if unique_matched_classes_per_run:
            print(f"   Total unique classes found: {len(unique_matched_classes_per_run)}")
            for class_name in sorted(unique_matched_classes_per_run):
                print(f"   - {class_name}")
        else:
            print("   No class matches found in any method")

        # Get class bodies for all matched classes
        print(f"\n📦 Extracting class bodies for matched classes...")
        matched_class_bodies = get_class_bodies_for_matched_classes(unique_matched_classes_per_run)

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

        # Save to separate files with K and sim values
        filename = f'./data/prompt_to_llm_K{K}_sim{sim}.txt'
        with open(filename, 'w') as f:
            f.write(final_prompt_to_llm)
        
        print(f"🧠 Final prompt saved to '{filename}'")
        print(f"📚 {len(unique_matched_classes_per_run)} unique class matches for this run")
        print(f"📋 {len(matched_class_bodies)} class bodies extracted and saved")
