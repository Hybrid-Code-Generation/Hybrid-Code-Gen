# Generic LLM Response Comparison Tool
# 
# This script compares LLM-generated code responses with a correct reference method.
# It performs functional correctness analysis by comparing:
# - Method signatures (return type, method name, parameters)
# - Method calls used within the code
# - Control structures (if/else, loops, try/catch, etc.)
# - Keywords and language constructs
# 
# To use this tool:
# 1. Replace the content of 'correct_method' variable below with your reference method
# 2. Ensure your LLM responses are in ./data/llm_responses/ as .md files
# 3. Run the script to get JSON results and summary statistics
#
# The tool will generate:
# - comparison_results.json: Detailed analysis for each file
# - Console output: Summary statistics and detailed JSON dump

correct_method = """
    public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
        Collection<Owner> owners;
        if (lastName != null) {
            owners = this.clinicService.findOwnerByLastName(lastName);
        } else {
            owners = this.clinicService.findAllOwners();
        }
        if (owners.isEmpty()) {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
        return new ResponseEntity<>(ownerMapper.toOwnerDtoCollection(owners), HttpStatus.OK);
    }
"""

import os
import logging
import json
import re
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
data_dir = "./data/llm_responses"

def extract_java_code(content: str) -> str:
    """Extract Java code from markdown content."""
    # Look for code blocks with java language specification
    java_code_pattern = r'```(?:java)?\n(.*?)```'
    matches = re.findall(java_code_pattern, content, re.DOTALL)
    
    if matches:
        # Return the first Java code block found
        return matches[0].strip()
    
    # If no code blocks found, return the content as is (might be plain Java)
    return content.strip()

def extract_method_components(code: str) -> Dict[str, Any]:
    """Extract key components from a method for comparison."""
    components = {
        "method_signature": "",
        "return_type": "",
        "method_name": "",
        "parameters": [],
        "method_calls": [],
        "keywords": [],
        "control_structures": [],
        "string_literals": [],
        "variable_declarations": []
    }
    
    # Clean code for analysis
    clean_code = re.sub(r'\s+', ' ', code.strip())
    
    # Extract method signature
    method_pattern = r'(public|private|protected)?\s*(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)'
    method_match = re.search(method_pattern, clean_code)
    if method_match:
        components["method_signature"] = method_match.group(0)
        components["return_type"] = method_match.group(2) if method_match.group(2) else ""
        components["method_name"] = method_match.group(3)
        
        # Extract parameters
        params = method_match.group(4)
        if params.strip():
            param_list = [p.strip() for p in params.split(',')]
            components["parameters"] = param_list
    
    # Extract method calls (words followed by parentheses)
    method_calls = re.findall(r'(\w+)\s*\(', clean_code)
    components["method_calls"] = list(set(method_calls))
    
    # Extract keywords and identifiers
    keywords = re.findall(r'\b(if|else|while|for|try|catch|return|new|this|null|true|false)\b', clean_code)
    components["keywords"] = list(set(keywords))
    
    # Extract control structures
    control_patterns = [
        (r'if\s*\([^)]+\)', 'if_statement'),
        (r'else', 'else_statement'),
        (r'while\s*\([^)]+\)', 'while_loop'),
        (r'for\s*\([^)]+\)', 'for_loop'),
        (r'try\s*\{', 'try_block'),
        (r'catch\s*\([^)]+\)', 'catch_block')
    ]
    for pattern, structure_type in control_patterns:
        if re.search(pattern, clean_code):
            components["control_structures"].append(structure_type)
    
    # Extract string literals
    string_literals = re.findall(r'"([^"]*)"', clean_code)
    components["string_literals"] = string_literals
    
    # Extract variable declarations (simple pattern)
    var_declarations = re.findall(r'(\w+(?:<[^>]+>)?)\s+(\w+)\s*[=;]', clean_code)
    components["variable_declarations"] = var_declarations
    
    return components

def analyze_functional_correctness(llm_code: str, correct_code: str) -> Dict[str, Any]:
    """Generic functional correctness analysis based on code structure comparison."""
    
    results = {
        "method_signature_match": False,
        "return_type_match": False,
        "method_name_match": False,
        "parameter_match": False,
        "method_calls_coverage": 0.0,
        "control_structure_coverage": 0.0,
        "keyword_coverage": 0.0,
        "overall_functional_score": 0.0,
        "detailed_analysis": {},
        "issues_found": []
    }
    
    # Extract components from both code snippets
    llm_components = extract_method_components(llm_code)
    correct_components = extract_method_components(correct_code)
    
    # Store detailed analysis
    results["detailed_analysis"] = {
        "llm_components": llm_components,
        "correct_components": correct_components
    }
    
    # 1. Method signature comparison
    if llm_components["method_signature"] and correct_components["method_signature"]:
        # Normalize signatures for comparison
        llm_sig = re.sub(r'\s+', ' ', llm_components["method_signature"]).strip()
        correct_sig = re.sub(r'\s+', ' ', correct_components["method_signature"]).strip()
        
        # Check similarity (allowing for minor differences in modifiers/formatting)
        llm_sig_parts = llm_sig.split()
        correct_sig_parts = correct_sig.split()
        
        if len(llm_sig_parts) >= 2 and len(correct_sig_parts) >= 2:
            # Compare return type and method name
            results["return_type_match"] = llm_components["return_type"] == correct_components["return_type"]
            results["method_name_match"] = llm_components["method_name"] == correct_components["method_name"]
            results["parameter_match"] = llm_components["parameters"] == correct_components["parameters"]
            
            if results["return_type_match"] and results["method_name_match"] and results["parameter_match"]:
                results["method_signature_match"] = True
            else:
                if not results["return_type_match"]:
                    results["issues_found"].append(f"Return type mismatch: expected '{correct_components['return_type']}', got '{llm_components['return_type']}'")
                if not results["method_name_match"]:
                    results["issues_found"].append(f"Method name mismatch: expected '{correct_components['method_name']}', got '{llm_components['method_name']}'")
                if not results["parameter_match"]:
                    results["issues_found"].append(f"Parameter mismatch: expected {correct_components['parameters']}, got {llm_components['parameters']}")
    
    # 2. Method calls coverage
    if correct_components["method_calls"]:
        llm_calls = set(llm_components["method_calls"])
        correct_calls = set(correct_components["method_calls"])
        
        found_calls = llm_calls.intersection(correct_calls)
        results["method_calls_coverage"] = len(found_calls) / len(correct_calls)
        
        missing_calls = correct_calls - llm_calls
        if missing_calls:
            results["issues_found"].append(f"Missing method calls: {list(missing_calls)}")
    
    # 3. Control structure coverage
    if correct_components["control_structures"]:
        llm_structures = set(llm_components["control_structures"])
        correct_structures = set(correct_components["control_structures"])
        
        found_structures = llm_structures.intersection(correct_structures)
        results["control_structure_coverage"] = len(found_structures) / len(correct_structures)
        
        missing_structures = correct_structures - llm_structures
        if missing_structures:
            results["issues_found"].append(f"Missing control structures: {list(missing_structures)}")
    
    # 4. Keyword coverage
    if correct_components["keywords"]:
        llm_keywords = set(llm_components["keywords"])
        correct_keywords = set(correct_components["keywords"])
        
        found_keywords = llm_keywords.intersection(correct_keywords)
        results["keyword_coverage"] = len(found_keywords) / len(correct_keywords)
    
    # Calculate overall functional score
    scores = [
        results["method_signature_match"],
        results["method_calls_coverage"],
        results["control_structure_coverage"],
        results["keyword_coverage"]
    ]
    
    # Weight the scores (method signature and calls are more important)
    weights = [0.4, 0.3, 0.2, 0.1]
    weighted_score = sum(score * weight for score, weight in zip(scores, weights))
    
    results["overall_functional_score"] = weighted_score
    
    return results

def compare_with_similarity(llm_code: str, correct_code: str) -> float:
    """Calculate basic text similarity score."""
    llm_words = set(re.findall(r'\w+', llm_code.lower()))
    correct_words = set(re.findall(r'\w+', correct_code.lower()))
    
    if not correct_words:
        return 0.0
    
    intersection = llm_words.intersection(correct_words)
    return len(intersection) / len(correct_words)

# Main comparison logic
md_files = [f for f in os.listdir(data_dir) if f.endswith('.md') and os.path.isfile(os.path.join(data_dir, f))]

logger.info(f"Found {len(md_files)} md files to compare")

comparison_results = {}

for md_file in md_files:
    file_path = os.path.join(data_dir, md_file)
    logger.info(f"Comparing file: {md_file}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract Java code from the markdown content
    llm_code = extract_java_code(content)
    
    # Perform functional correctness analysis
    functional_analysis = analyze_functional_correctness(llm_code, correct_method)
    
    # Calculate similarity score
    similarity_score = compare_with_similarity(llm_code, correct_method)
      # Store results
    comparison_results[md_file] = {
        "file_name": md_file,
        "extracted_code": llm_code,
        "functional_analysis": functional_analysis,
        "similarity_score": similarity_score,
        "summary": {
            "functional_score": functional_analysis["overall_functional_score"],
            "similarity_score": similarity_score,
            "issues_count": len(functional_analysis["issues_found"]),
            "is_functionally_correct": functional_analysis["overall_functional_score"] >= 0.7,
            "method_signature_match": functional_analysis["method_signature_match"],
            "method_calls_coverage": functional_analysis["method_calls_coverage"],
            "control_structure_coverage": functional_analysis["control_structure_coverage"]
        }
    }
    
    logger.info(f"Functional score: {functional_analysis['overall_functional_score']:.2f}, Similarity: {similarity_score:.2f}")

# Save results to JSON file
output_file = "comparison_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)

logger.info(f"Results saved to {output_file}")

# Print summary for each method
print("\nCOMPARISON RESULTS SUMMARY:")
print("-" * 50)

for file_name, result in comparison_results.items():
    summary = result["summary"]
    status = "PASS" if summary["is_functionally_correct"] else "FAIL"
    print(f"{file_name}: {status} (Score: {summary['functional_score']:.2f})")
