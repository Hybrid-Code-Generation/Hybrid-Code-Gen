# This file is supposed to create a methods.csv file for a given java repo and then generate embeddings for the code chunks using Azure OpenAI

# Code for creating methods.csv and Class.csv
from AST.java_parser_working import process_directory, write_to_csv, write_classes_to_csv

method_infos = []
class_infos = []
directory_path = "C:\\repos\\Hybrid-Code-Gen\\javarepoparser\\temp\\JavaBench\\projects\\PA19"
output_file = "methods.csv"
class_output_file = "Class.csv"

process_directory(directory_path, method_infos, class_infos)
write_to_csv(output_file, method_infos)
write_classes_to_csv(class_output_file, class_infos)

print(f"Method information written to {output_file}")
print(f"Class information written to {class_output_file}")

# Code for generating embeddings and saving it in pickle file
from code_embedding_clean_openai import generate_embeddings_and_save

generate_embeddings_and_save(csv_path=output_file, batch_size=20)