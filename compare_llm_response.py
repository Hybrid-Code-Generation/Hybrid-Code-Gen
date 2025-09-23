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
from openai import AzureOpenAI

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
from datetime import datetime
from typing import Dict, List, Any
from key import key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
data_dir = "./data/llm_responses"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

endpoint = "https://azure-ai-hackthon.openai.azure.com/"
model_name = "gpt-4.1"
deployment = "gpt-4.1"

subscription_key = key
api_version = "2024-12-01-preview"

logger.info("Initializing Azure OpenAI client")
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)



# Main comparison logic
md_files = [f for f in os.listdir(data_dir) if f.endswith('.md') and os.path.isfile(os.path.join(data_dir, f))]

logger.info(f"Found {len(md_files)} md files to compare")

comparison_results = {}

llm_prompt = """
You are an expert AI code reviewer named *CodeJudge*. Your task is to meticulously compare a {{GENERATED_METHOD}} against a {{CORRECT_METHOD}} and provide a functional similarity score out of 10.

You must adhere strictly to the *Objective Scoring Rubric* defined below. For each criterion, award points only if the {{GENERATED_METHOD}} fully and correctly implements the feature as seen in the {{CORRECT_METHOD}}. There is no partial credit for a criterion.

-----

## *Objective Scoring Rubric (10 points total)*

*1. Method Signature (2 Points)*

  * *(1 pt) - Interface Match:* The method name, parameter types, and return type are identical to the {{CORRECT_METHOD}}.
  * *(1 pt) - Declaration Match:* The access modifier (e.g., public) and any essential annotations (e.g., @Override) are identical to the {{CORRECT_METHOD}}.

*2. Core Functional Logic (3 Points)*

  * *(3 pts) - "Happy Path" Equivalence:* The primary logic, including conditional branches (if/else structures) and service/method calls for the main success scenario, functions identically to the {{CORRECT_METHOD}}. The generated method must produce the same successful outcome when given a standard, non-edge-case input.

*3. Critical Edge Case Handling (3 Points)*

  * *(3 pts) - Robustness Match:* The method correctly implements the specific edge case handling shown in the {{CORRECT_METHOD}}. This is most commonly a check for null inputs or an empty result set that triggers a different return path (e.g., returning a 404 NOT FOUND status instead of a 200 OK). If the {{GENERATED_METHOD}} fails to replicate this specific behavior, it gets 0 points for this criterion.

*4. Architectural Compliance (2 Points)*

  * *(2 pts) - Abstraction and Tooling:* The method correctly uses the same helper classes, mappers, services, or other abstractions that are used in the {{CORRECT_METHOD}} (e.g., using a dedicated ownerMapper instead of re-implementing the mapping logic inline). It must not bypass the intended architecture.

-----

## *Input Methods*

{correct_method}

java
// Paste the correct method here


{generated_method}

java
// Paste the generated method to be evaluated here


-----

## *Required Output Format*

Provide your response in the following format:

*Similarity Score:* X / 10

*Verdict:* A brief, one-sentence summary of the evaluation (e.g., "Functionally close but critically flawed," or "Architecturally divergent and functionally incorrect.")

### *Reasoning*

  * *Method Signature:* (Score) - Justification.
  * *Core Functional Logic:* (Score) - Justification.
  * *Critical Edge Case Handling:* (Score) - Justification.
  * *Architectural Compliance:* (Score) - Justification.
"""

for md_file in md_files:
    file_path = os.path.join(data_dir, md_file)
    logger.info(f"Processing file: {md_file}")
    
    try:
        # Read the MD file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract Java code from the MD file (assuming it's in code blocks)
        java_code_pattern = r'```java\s*(.*?)\s*```'
        java_matches = re.findall(java_code_pattern, content, re.DOTALL)
        
        if not java_matches:
            # Try without language specification
            code_pattern = r'```\s*(.*?)\s*```'
            code_matches = re.findall(code_pattern, content, re.DOTALL)
            generated_method = code_matches[0].strip() if code_matches else content.strip()
        else:
            generated_method = java_matches[0].strip()
        
        # Format the prompt with both methods
        formatted_prompt = llm_prompt.format(
            correct_method=correct_method.strip(),
            generated_method=generated_method
        )
        
        logger.info(f"Sending request to Azure OpenAI for {md_file}")
        
        # Send request to Azure OpenAI
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Java code evaluator named CodeJudge. Follow the evaluation format exactly as specified in the prompt."
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            max_completion_tokens=2000,
            temperature=0.1,  # Lower temperature for more consistent JSON output
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment
        )
        
        logger.info(f"Response received for {md_file}")
        response_content = response.choices[0].message.content
        
        # Parse the text-based response and extract structured data
        try:
            # Extract similarity score
            score_pattern = r'\*?Similarity Score:\*?\s*(\d+(?:\.\d+)?)\s*/\s*10'
            score_match = re.search(score_pattern, response_content, re.IGNORECASE)
            similarity_score = float(score_match.group(1)) if score_match else None
            
            # Extract verdict
            verdict_pattern = r'\*?Verdict:\*?\s*(.+?)(?=\n|$)'
            verdict_match = re.search(verdict_pattern, response_content, re.IGNORECASE)
            verdict = verdict_match.group(1).strip() if verdict_match else None
            
            # Extract individual criterion scores
            method_sig_pattern = r'\*?Method Signature:\*?\s*(\d+(?:\.\d+)?)\s*-\s*(.+?)(?=\n|\*|$)'
            core_logic_pattern = r'\*?Core Functional Logic:\*?\s*(\d+(?:\.\d+)?)\s*-\s*(.+?)(?=\n|\*|$)'
            edge_case_pattern = r'\*?Critical Edge Case Handling:\*?\s*(\d+(?:\.\d+)?)\s*-\s*(.+?)(?=\n|\*|$)'
            arch_pattern = r'\*?Architectural Compliance:\*?\s*(\d+(?:\.\d+)?)\s*-\s*(.+?)(?=\n|\*|$)'
            
            method_sig_match = re.search(method_sig_pattern, response_content, re.IGNORECASE | re.DOTALL)
            core_logic_match = re.search(core_logic_pattern, response_content, re.IGNORECASE | re.DOTALL)
            edge_case_match = re.search(edge_case_pattern, response_content, re.IGNORECASE | re.DOTALL)
            arch_match = re.search(arch_pattern, response_content, re.IGNORECASE | re.DOTALL)
            
            # Create structured evaluation result
            evaluation_result = {
                "similarity_score": similarity_score,
                "verdict": verdict,
                "detailed_scores": {
                    "method_signature": {
                        "score": float(method_sig_match.group(1)) if method_sig_match else None,
                        "justification": method_sig_match.group(2).strip() if method_sig_match else None
                    },
                    "core_functional_logic": {
                        "score": float(core_logic_match.group(1)) if core_logic_match else None,
                        "justification": core_logic_match.group(2).strip() if core_logic_match else None
                    },
                    "critical_edge_case_handling": {
                        "score": float(edge_case_match.group(1)) if edge_case_match else None,
                        "justification": edge_case_match.group(2).strip() if edge_case_match else None
                    },
                    "architectural_compliance": {
                        "score": float(arch_match.group(1)) if arch_match else None,
                        "justification": arch_match.group(2).strip() if arch_match else None
                    }
                }
            }
            
            # Store the result
            comparison_results[md_file] = {
                "file_name": md_file,
                "generated_method": generated_method,
                "evaluation": evaluation_result,
                "raw_response": response_content,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed {md_file} - Score: {similarity_score}/10")
            
        except Exception as e:
            logger.error(f"Failed to parse response for {md_file}: {e}")
            comparison_results[md_file] = {
                "file_name": md_file,
                "generated_method": generated_method,
                "evaluation": None,
                "raw_response": response_content,
                "error": f"Parsing error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error processing {md_file}: {str(e)}")
        comparison_results[md_file] = {
            "file_name": md_file,
            "generated_method": None,
            "evaluation": None,
            "raw_response": None,
            "error": f"Processing error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Save results to JSON file
output_file = "comparison_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)

logger.info(f"Results saved to {output_file}")

# Print summary for each method
print("\nCOMPARISON RESULTS SUMMARY:")
print("-" * 50)

successful_evaluations = 0
failed_evaluations = 0

for file_name, result in comparison_results.items():
    if "error" in result:
        print(f"{file_name}: ERROR - {result['error']}")
        failed_evaluations += 1
    elif result.get("evaluation") is None:
        print(f"{file_name}: FAILED - No evaluation result")
        failed_evaluations += 1
    else:
        evaluation = result["evaluation"]
        if evaluation.get("similarity_score") is not None:
            similarity_score = evaluation["similarity_score"]
            verdict = evaluation.get("verdict", "No verdict provided")
            print(f"{file_name}: SUCCESS (Score: {similarity_score}/10) - {verdict}")
            successful_evaluations += 1
        else:
            print(f"{file_name}: PARTIAL - Evaluation completed but missing similarity score")
            successful_evaluations += 1

print(f"\nSUMMARY: {successful_evaluations} successful, {failed_evaluations} failed out of {len(comparison_results)} total files")

# Print detailed JSON results
print(f"\nDetailed results saved to {output_file}")
print("\nSample result structure:")
if comparison_results:
    sample_key = next(iter(comparison_results))
    sample_result = comparison_results[sample_key]
    print(json.dumps({sample_key: sample_result}, indent=2)[:500] + "..." if len(str(sample_result)) > 500 else json.dumps({sample_key: sample_result}, indent=2))
