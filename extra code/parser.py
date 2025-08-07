import os
import csv
from tree_sitter import Language, Parser

# Load the C# language grammar (You may need to compile it first)
CSHARP_LANGUAGE = Language('tree_sitter_c_sharp')

# Function to parse a C# file and extract method details
def parse_csharp_code(file_path, method_infos):
    parser = Parser()
    parser.set_language(CSHARP_LANGUAGE)

    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()

    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    extract_methods(file_path, root_node, method_infos)

# Function to extract method information from syntax tree
def extract_methods(file_path, node, method_infos, namespace=None, class_name=None):
    if node.type == 'namespace_declaration':
        namespace = get_node_text(node.child_by_field_name('name'))

    if node.type == 'class_declaration':
        class_name = get_node_text(node.child_by_field_name('name'))

    if node.type == 'method_declaration':
        method_name = get_node_text(node.child_by_field_name('name'))
        return_type = get_node_text(node.child_by_field_name('type'))
        parameters = extract_parameters(node.child_by_field_name('parameter_list'))
        function_body = get_node_text(node.child_by_field_name('body'))

        method_infos.append({
            "FilePath": file_path,
            "Namespace": namespace or "",
            "Class": class_name or "",
            "Method Name": method_name or "",
            "Return Type": return_type or "",
            "Parameters": parameters or "",
            "Function Body": function_body.strip() if function_body else ""
        })

    for child in node.children:
        extract_methods(file_path, child, method_infos, namespace, class_name)

# Helper function to extract text from a syntax node
def get_node_text(node):
    return node.text.decode('utf-8') if node else ""

# Extract parameter list from method declaration
def extract_parameters(param_node):
    if not param_node:
        return ""
    params = []
    for param in param_node.children:
        if param.type == 'parameter':
            param_type = get_node_text(param.child_by_field_name('type'))
            param_name = get_node_text(param.child_by_field_name('name'))
            params.append(f"{param_type} {param_name}")
    return ", ".join(params)

# Function to process all C# files recursively in a directory
def process_directory(directory_path, method_infos):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".cs"):
                file_path = os.path.join(root, file)
                print(f"Parsing file: {file_path}")
                parse_csharp_code(file_path, method_infos)

# Function to write method info to a CSV file
def write_to_csv(output_file, method_infos):
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["FilePath", "Namespace", "Class", "Method Name", "Return Type", "Parameters", "Function Body"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for info in method_infos:
            writer.writerow(info)

# Main execution
if __name__ == "__main__":
    directory_path = r"C:\Users\divchauhan\EnhancedRestore"  # Replace with the actual directory path
    output_file = "test.csv"
    method_infos = [] 

    process_directory(directory_path, method_infos)
    write_to_csv(output_file, method_infos)

    print(f"Output written to {output_file}")
