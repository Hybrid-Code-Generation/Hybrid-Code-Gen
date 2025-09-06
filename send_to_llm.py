import os
import logging
from datetime import datetime
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

endpoint = "https://azure-ai-hackthon.openai.azure.com/"
model_name = "gpt-4.1"
deployment = "gpt-4.1"

subscription_key = ""
api_version = "2024-12-01-preview"

logger.info("Initializing Azure OpenAI client")
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

system_prompt = """
You are a code generation assistant. The workflow is solely for generating code based on retrieved context. Focus on generate the requested code using the provided context. Only provide the code in your response, do not include any explanations or additional text.
"""

# Create output directory if it doesn't exist
output_dir = "./data/llm_responses"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Output directory: {output_dir}")

# Get all txt files in ./data/ folder
data_dir = "./data/"
txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt') and os.path.isfile(os.path.join(data_dir, f))]
logger.info(f"Found {len(txt_files)} txt files to process")

for txt_file in txt_files:
    file_path = os.path.join(data_dir, txt_file)
    logger.info(f"Processing file: {txt_file}")
    
    # Read prompt from file
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    logger.info(f"Prompt loaded from {txt_file}, length: {len(prompt)} characters")
    
    # Send request to Azure OpenAI
    logger.info(f"Sending request to Azure OpenAI for {txt_file}")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_completion_tokens=13107,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )
    
    logger.info(f"Response received for {txt_file}")
    response_content = response.choices[0].message.content
    
    # Save response to markdown file with same base name
    base_name = os.path.splitext(txt_file)[0]
    md_filename = os.path.join(output_dir, f"{base_name}.md")
    
    logger.info(f"Saving response to: {md_filename}")
    with open(md_filename, 'w', encoding='utf-8') as md_file:
        md_file.write(response_content)
    
    logger.info(f"Response saved successfully, length: {len(response_content)} characters")

logger.info(f"All files processed. Responses saved in: {output_dir}")