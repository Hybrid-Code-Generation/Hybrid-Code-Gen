import os
import csv
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

# Method 1: Using tree-sitter-languages (Recommended - easiest)
try:
    JAVA_LANGUAGE = Language(tsjava.language())
    print("Successfully loaded Java language using tree-sitter-languages")
    print("JAVA_LANGUAGE is:", JAVA_LANGUAGE)
except ImportError:
    print("tree-sitter-languages not found. Install with: pip install tree-sitter-languages")
    JAVA_LANGUAGE = None

# Method 2: Fallback to manual loading (if you have a pre-built .so/.dll file)
if JAVA_LANGUAGE is None:
    try:
        from tree_sitter import Language
        # Fixed: Language constructor takes only one argument (path to shared library)
        JAVA_LANGUAGE = Language('tree_sitter_java.dll')  # or .so on Linux, .dylib on Mac
        print("Successfully loaded Java language from shared library")
    except Exception as e:
        print(f"Failed to load Java language: {e}")
        print("Please install tree-sitter-languages or build the grammar manually")
        exit(1)

def parse_java_code(file_path, method_infos):
    """Parse a Java file and extract method information"""
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()

        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        extract_methods(file_path, root_node, method_infos)
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")

def extract_methods(file_path, node, method_infos, package=None, class_name=None):
    """Recursively extract method information from AST nodes"""
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

    # Recursively process child nodes
    for child in node.children:
        extract_methods(file_path, child, method_infos, package, class_name)

def get_node_text(node):
    """Get text content from a node"""
    return node.text.decode('utf-8') if node else ""

def get_child_of_type(node, type_name):
    """Find first child node of specified type"""
    for child in node.children:
        if child.type == type_name:
            return child
    return None

def extract_parameters(param_node):
    """Extract parameter information from parameter node"""
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
    """Extract method modifiers (public, private, static, etc.)"""
    modifiers = []
    for child in node.children:
        if child.type == 'modifiers':
            for mod in child.children:
                if mod.type != ',':
                    modifiers.append(get_node_text(mod))
    return " ".join(modifiers)

def process_directory(directory_path, method_infos):
    """Process all Java files in a directory recursively"""
    java_files_count = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                print(f"Parsing file: {file_path}")
                parse_java_code(file_path, method_infos)
                java_files_count += 1
    
    print(f"Processed {java_files_count} Java files")
    print(f"Found {len(method_infos)} methods")

def write_to_csv(output_file, method_infos):
    """Write method information to CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["FilePath", "Package", "Class", "Method Name", "Return Type",
                      "Parameters", "Function Body", "Throws", "Modifiers", "Generics"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for info in method_infos:
            writer.writerow(info)

if __name__ == "__main__":
    # Configuration
    directory_path = r"C:\Users\divchauhan\Downloads\Library-Assistant-master\Library-Assistant-master"  # Replace with your Java code directory
    output_file = "java_parsed.csv"
    method_infos = []

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist!")
        exit(1)

    print(f"Starting to process Java files in: {directory_path}")
    
    # Process all Java files
    process_directory(directory_path, method_infos)
    
    # Write results to CSV
    write_to_csv(output_file, method_infos)

    print(f"Output written to {output_file}")
    print(f"Total methods extracted: {len(method_infos)}")