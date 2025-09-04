import os
import csv
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
import os
import csv

# Load the Java grammar (ensure it was compiled into a .so/.dll/.dylib file)
JAVA_LANGUAGE = get_language('java')

def parse_java_code(file_path, method_infos, class_infos=None):
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()

    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    extract_methods(file_path, root_node, method_infos)
    
    # Extract class information if class_infos list is provided
    if class_infos is not None:
        extract_classes(file_path, root_node, class_infos)

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

def extract_classes(file_path, node, class_infos, package=None):
    """Extract class information from the AST"""
    if node.type == 'package_declaration':
        package = get_node_text(node.child_by_field_name('name'))

    if node.type in ('class_declaration', 'interface_declaration'):
        class_name = get_node_text(node.child_by_field_name('name'))
        class_body = get_class_body_text(node)
        
        class_infos.append({
            "FilePath": file_path,
            "Package": package or "",
            "Class": class_name or "",
            "ClassBody": class_body.strip() if class_body else ""
        })

    for child in node.children:
        extract_classes(file_path, child, class_infos, package)

def get_class_body_text(node):
    """Extract the body of a class or interface"""
    body_node = node.child_by_field_name('body')
    if body_node:
        return get_node_text(body_node)
    return ""

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

def process_directory(directory_path, method_infos, class_infos=None):
    print(f"Processing directory: {directory_path}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                print(f"Parsing file: {file_path}")
                parse_java_code(file_path, method_infos, class_infos)

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

def write_classes_to_csv(output_file, class_infos):
    """Write class information to CSV file"""
    # delete existing file
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["FilePath", "Package", "Class", "ClassBody"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for info in class_infos:
            writer.writerow(info)

def extract_classes_only(directory_path, output_file="Class.csv"):
    """Convenience function to extract only class information"""
    class_infos = []
    process_directory(directory_path, [], class_infos)
    write_classes_to_csv(output_file, class_infos)
    print(f"Class information written to {output_file}")
    return class_infos

def extract_methods_and_classes(directory_path, method_output_file="java_parsed.csv", class_output_file="Class.csv"):
    """Convenience function to extract both method and class information"""
    method_infos = []
    class_infos = []
    process_directory(directory_path, method_infos, class_infos)
    write_to_csv(method_output_file, method_infos)
    write_classes_to_csv(class_output_file, class_infos)
    print(f"Method information written to {method_output_file}")
    print(f"Class information written to {class_output_file}")
    return method_infos, class_infos

# if __name__ == "__main__":
#     directory_path = r"C:\Users\divchauhan\Downloads\Library-Assistant-master\Library-Assistant-master"  # Replace with Java code dir
#     method_output_file = "java_parsed.csv"
#     class_output_file = "Class.csv"
#     method_infos = []
#     class_infos = []
#
#     # Process directory for both methods and classes
#     process_directory(directory_path, method_infos, class_infos)
#     
#     # Write method information to CSV (existing functionality)
#     write_to_csv(method_output_file, method_infos)
#     
#     # Write class information to CSV (new functionality)
#     write_classes_to_csv(class_output_file, class_infos)
#
#     print(f"Method output written to {method_output_file}")
#     print(f"Class output written to {class_output_file}")
