import os
import csv
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language

# Load the Java grammar (ensure it was compiled into a .so/.dll/.dylib file)
JAVA_LANGUAGE = get_language('java')

def parse_java_code(file_path, method_infos):
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()

    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    extract_methods(file_path, root_node, method_infos)

def extract_methods(file_path, node, method_infos, package=None, class_name=None):
    if node.type == 'package_declaration':
        package = get_node_text(node.child_by_field_name('name'))

    if node.type in ('class_declaration', 'interface_declaration'):
        class_name = get_node_text(node.child_by_field_name('name'))

    if node.type == 'method_declaration':
        method_name = get_node_text(node.child_by_field_name('name'))
        return_type = get_node_text(node.child_by_field_name('type'))
        parameters = extract_parameters(node.child_by_field_name('parameters'))
        body = get_node_text(node.child_by_field_name('body'))
        throws_clause = get_node_text(get_child_of_type(node, 'throws'))
        modifiers = get_modifiers(node)
        generics = get_node_text(node.child_by_field_name('type_parameters'))

        method_infos.append({
            "FilePath": file_path,
            "Package": package or "",
            "Class": class_name or "",
            "Method Name": method_name or "",
            "Return Type": return_type or "",
            "Parameters": parameters or "",
            "Function Body": body.strip() if body else "",
            "Throws": throws_clause or "",
            "Modifiers": modifiers or "",
            "Generics": generics or ""
        })

    for child in node.children:
        extract_methods(file_path, child, method_infos, package, class_name)

def get_node_text(node):
    return node.text.decode('utf-8') if node else ""

def get_child_of_type(node, type_name):
    for child in node.children:
        if child.type == type_name:
            return child
    return None

def extract_parameters(param_node):
    if not param_node:
        return ""
    params = []
    for param in param_node.children:
        if param.type == 'formal_parameter':
            param_type = get_node_text(param.child_by_field_name('type'))
            param_name = get_node_text(param.child_by_field_name('name'))
            params.append(f"{param_type} {param_name}")
    return ", ".join(params)

def get_modifiers(node):
    modifiers = []
    for child in node.children:
        if child.type == 'modifiers':
            for mod in child.children:
                if mod.type != ',':
                    modifiers.append(get_node_text(mod))
    return " ".join(modifiers)

def process_directory(directory_path, method_infos):
    print(f"Processing directory: {directory_path}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                print(f"Parsing file: {file_path}")
                parse_java_code(file_path, method_infos)

def write_to_csv(output_file, method_infos):
    # delete existing file
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["FilePath", "Package", "Class", "Method Name", "Return Type",
                      "Parameters", "Function Body", "Throws", "Modifiers", "Generics"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for info in method_infos:
            writer.writerow(info)

# if __name__ == "__main__":
#     directory_path = r"C:\Users\divchauhan\Downloads\Library-Assistant-master\Library-Assistant-master"  # Replace with Java code dir
#     output_file = "java_parsed.csv"
#     method_infos = []

#     process_directory(directory_path, method_infos)
#     write_to_csv(output_file, method_infos)

#     print(f"Output written to {output_file}")
