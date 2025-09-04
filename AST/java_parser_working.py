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
    """Extract the body of a class or interface with method declarations only (no method bodies)"""
    body_node = node.child_by_field_name('body')
    if not body_node:
        return ""
    
    # Build the class body with method signatures but not method bodies
    class_content = []
    class_content.append("{")
    
    # Extract fields, method declarations, constructors, etc.
    for child in body_node.children:
        if child.type == 'field_declaration':
            # Include full field declarations
            class_content.append("    " + get_node_text(child).strip())
        elif child.type == 'method_declaration':
            # Include only method signature, not body
            method_signature = get_method_signature(child)
            class_content.append("    " + method_signature)
        elif child.type == 'constructor_declaration':
            # Include only constructor signature, not body
            constructor_signature = get_constructor_signature(child)
            class_content.append("    " + constructor_signature)
        elif child.type == 'class_declaration' or child.type == 'interface_declaration':
            # Include nested classes/interfaces (you might want to handle these differently)
            nested_name = get_node_text(child.child_by_field_name('name'))
            class_content.append(f"    // Nested class: {nested_name}")
        elif child.type == 'enum_declaration':
            # Include enum declarations
            enum_name = get_node_text(child.child_by_field_name('name'))
            class_content.append(f"    enum {enum_name} {{ /* enum body */ }}")
    
    class_content.append("}")
    return "\n".join(class_content)

def get_node_text(node):
    return node.text.decode('utf-8') if node else ""

def get_method_signature(method_node):
    """Extract method signature without body"""
    modifiers = get_modifiers(method_node)
    generics = ""
    
    # Get generics if present
    type_params = get_child_of_type(method_node, 'type_parameters')
    if type_params:
        generics = get_node_text(type_params)
    
    # Get return type
    return_type = get_node_text(method_node.child_by_field_name('type'))
    
    # Get method name
    method_name = get_node_text(method_node.child_by_field_name('name'))
    
    # Get parameters
    param_node = method_node.child_by_field_name('parameters')
    parameters = extract_parameters(param_node)
    
    # Get throws clause
    throws_clause = ""
    for child in method_node.children:
        if child.type == 'throws':
            throws_clause = get_node_text(child)
            break
    
    # Build signature
    signature_parts = []
    if modifiers:
        signature_parts.append(modifiers)
    if generics:
        signature_parts.append(generics)
    if return_type:
        signature_parts.append(return_type)
    if method_name:
        signature_parts.append(method_name)
    
    signature = " ".join(signature_parts) + f"({parameters})"
    if throws_clause:
        signature += " " + throws_clause
    signature += ";"
    
    return signature

def get_constructor_signature(ctor_node):
    """Extract constructor signature without body"""
    modifiers = get_modifiers(ctor_node)
    
    # Get constructor name
    ctor_name = get_node_text(ctor_node.child_by_field_name('name'))
    
    # Get parameters
    param_node = ctor_node.child_by_field_name('parameters')
    parameters = extract_parameters(param_node)
    
    # Get throws clause
    throws_clause = ""
    for child in ctor_node.children:
        if child.type == 'throws':
            throws_clause = get_node_text(child)
            break
    
    # Build signature
    signature_parts = []
    if modifiers:
        signature_parts.append(modifiers)
    if ctor_name:
        signature_parts.append(ctor_name)
    
    signature = " ".join(signature_parts) + f"({parameters})"
    if throws_clause:
        signature += " " + throws_clause
    signature += ";"
    
    return signature

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
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for info in method_infos:
            # Clean the data to avoid CSV issues
            cleaned_info = {}
            for key, value in info.items():
                if value is None:
                    cleaned_info[key] = ""
                else:
                    # Replace problematic characters
                    cleaned_value = str(value).replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
                    cleaned_info[key] = cleaned_value
            writer.writerow(cleaned_info)

def write_classes_to_csv(output_file, class_infos):
    """Write class information to CSV file"""
    # delete existing file
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["FilePath", "Package", "Class", "ClassBody"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for info in class_infos:
            # Clean the data to avoid CSV issues
            cleaned_info = {}
            for key, value in info.items():
                if value is None:
                    cleaned_info[key] = ""
                else:
                    # Replace problematic characters
                    cleaned_value = str(value).replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
                    cleaned_info[key] = cleaned_value
            writer.writerow(cleaned_info)

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
