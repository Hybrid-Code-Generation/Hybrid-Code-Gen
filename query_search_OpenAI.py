import re
import pandas as pd
import numpy as np
import faiss
from openai import AzureOpenAI
import nltk
from nltk.stem import WordNetLemmatizer

class CodeSearcher:
    def get_code_embedding(self, code_text: str) -> np.ndarray:
        """
        Generate an embedding for a code snippet or method body using Azure OpenAI.
        """
        response = self.client.embeddings.create(
            model=self.deployment,
            input=[code_text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    def __init__(self):
        self.client = None
        self.df = None
        self.index = None
        self.lemmatizer = None
        self.stopwords = {"the", "a", "an", "is", "to", "for", "of", "and", "in", "on", "by", "with", "from"}
        self.deployment = "text-embedding-3-large"
        
        # Data structure to store non-primitive types found across queries
        self.discovered_types = {
            'classes': set(),           # Custom classes
            'interfaces': set(),        # Interfaces
            'collections': set(),       # List, Set, Map, etc.
            'generics': set(),          # Generic type parameters
            'annotations': set(),       # @Annotations
            'enums': set()              # Enum types
        }
        
        # Java primitive types for filtering
        self.primitive_types = {
            'byte', 'short', 'int', 'long', 'float', 'double', 'boolean', 'char',
            'void', 'String'  # String is technically not primitive but commonly treated as such
        }
        
    def initialize(self):
        """Initialize Azure OpenAI client, load data, and setup NLP tools"""
        # Azure OpenAI Configuration
        endpoint = "https://azure-ai-hackthon.openai.azure.com/"
        api_version = "2024-12-01-preview"
        key = ""
        
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_deployment=self.deployment,
            api_key=key
        )
        print("✅ Azure OpenAI client initialized!")
        
        # Load FAISS index + metadata
        print("📄 Loading metadata and FAISS index...")
        self.df = pd.read_pickle("code_metadata.pkl")
        self.index = faiss.read_index("code_embeddings.index")
        print(f"✅ Loaded {len(self.df)} methods and FAISS index.")
        
        # Setup NLP tools
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_query_advanced(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r'[^a-z0-9_ ]', ' ', query)
        words = query.split()
        words = [w for w in words if w not in self.stopwords]
        words = [re.sub(r'([a-z])([A-Z])', r'\1 \2', w) for w in words]
        words = [w.replace('_', ' ') for w in words]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return " ".join(words[:30])
    
    def clean_query(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r'[^a-z0-9\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def extract_non_primitive_types(self, method_info):
        """
        Extract non-primitive data types from method parameters, return type, and body
        """
        non_primitive_types = {
            'classes': set(),
            'interfaces': set(),
            'collections': set(),
            'generics': set(),
            'annotations': set(),
            'enums': set()
        }
        
        # Extract from return type
        if method_info.get('Return Type'):
            self._parse_type_string(method_info['Return Type'], non_primitive_types)
        
        # Extract from parameters
        if method_info.get('Parameters'):
            self._parse_parameters(method_info['Parameters'], non_primitive_types)
        
        # Extract from method body
        if method_info.get('Function Body'):
            self._parse_method_body(method_info['Function Body'], non_primitive_types)
        
        # Update global discovered types
        for category, types in non_primitive_types.items():
            self.discovered_types[category].update(types)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in non_primitive_types.items()}
    
    def _parse_type_string(self, type_string, non_primitive_types):
        """Parse a type string and extract non-primitive types"""
        if not type_string or type_string.strip() in self.primitive_types:
            return
        
        # Remove array brackets
        type_string = re.sub(r'\[\]', '', type_string)
        
        # Handle generics like List<String>, Map<String, Integer>
        generic_match = re.findall(r'(\w+)<([^>]+)>', type_string)
        if generic_match:
            for outer_type, inner_types in generic_match:
                if outer_type not in self.primitive_types:
                    # Check if it's a common collection
                    if outer_type in ['List', 'Set', 'Map', 'ArrayList', 'LinkedList', 'HashSet', 'HashMap', 'TreeMap']:
                        non_primitive_types['collections'].add(outer_type)
                    else:
                        non_primitive_types['classes'].add(outer_type)
                
                # Parse inner types
                inner_type_list = [t.strip() for t in inner_types.split(',')]
                for inner_type in inner_type_list:
                    self._parse_type_string(inner_type, non_primitive_types)
        else:
            # Simple type without generics
            clean_type = type_string.strip()
            if clean_type and clean_type not in self.primitive_types:
                # Check if it's a collection type
                if clean_type in ['List', 'Set', 'Map', 'ArrayList', 'LinkedList', 'HashSet', 'HashMap', 'TreeMap']:
                    non_primitive_types['collections'].add(clean_type)
                else:
                    non_primitive_types['classes'].add(clean_type)
    
    def _parse_parameters(self, parameters_string, non_primitive_types):
        """Parse parameter string and extract types"""
        if not parameters_string:
            return
        
        # Better parameter parsing that handles generics properly
        params = []
        current_param = ""
        angle_count = 0
        
        for char in parameters_string:
            if char == '<':
                angle_count += 1
            elif char == '>':
                angle_count -= 1
            elif char == ',' and angle_count == 0:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
                continue
            current_param += char
        
        if current_param.strip():
            params.append(current_param.strip())
        
        # Extract type from each parameter
        for param in params:
            # Parameter format is usually "Type paramName" or "final Type paramName"
            parts = param.strip().split()
            if len(parts) >= 2:
                # Skip modifiers like 'final', 'static', etc.
                type_part = None
                for part in parts[:-1]:  # Last part is parameter name
                    if part not in ['final', 'static', 'public', 'private', 'protected']:
                        type_part = part
                        break
                
                if type_part:
                    self._parse_type_string(type_part, non_primitive_types)
    
    def _parse_method_body(self, method_body, non_primitive_types):
        """Parse method body and extract non-primitive types"""
        if not method_body:
            return
        
        # Extract class instantiations (new ClassName())
        new_instances = re.findall(r'new\s+(\w+(?:<[^>]+>)?)\s*\(', method_body)
        for instance in new_instances:
            self._parse_type_string(instance, non_primitive_types)
        
        # Extract static method calls (ClassName.methodName) - only capitalized class names
        static_calls = re.findall(r'([A-Z]\w+)\.(?:\w+)\s*\(', method_body)
        for class_name in static_calls:
            if class_name not in self.primitive_types:
                non_primitive_types['classes'].add(class_name)
        
        # Extract annotations (@AnnotationName)
        annotations = re.findall(r'@(\w+)', method_body)
        for annotation in annotations:
            non_primitive_types['annotations'].add(annotation)
        
        # Extract cast operations ((ClassName) variable) - only single word types
        casts = re.findall(r'\(\s*([A-Z]\w+(?:<[^>]+>)?)\s*\)', method_body)
        for cast_type in casts:
            self._parse_type_string(cast_type, non_primitive_types)
        
        # Extract instanceof checks (variable instanceof ClassName)
        instanceof_checks = re.findall(r'instanceof\s+([A-Z]\w+)', method_body)
        for type_name in instanceof_checks:
            if type_name not in self.primitive_types:
                non_primitive_types['classes'].add(type_name)
        
        # Extract variable declarations (ClassName variableName)
        variable_declarations = re.findall(r'\b([A-Z]\w+(?:<[^>]+>)?)\s+\w+\s*[=;]', method_body)
        for var_type in variable_declarations:
            self._parse_type_string(var_type, non_primitive_types)
    
    def get_query_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.deployment,
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    def get_discovered_types(self, category=None):
        """
        Get discovered non-primitive types
        Args:
            category: Optional category filter ('classes', 'interfaces', 'collections', 'generics', 'annotations', 'enums')
        Returns:
            Dictionary of discovered types or specific category if specified
        """
        if category:
            return list(self.discovered_types.get(category, set()))
        return {k: list(v) for k, v in self.discovered_types.items()}
    
    def clear_discovered_types(self):
        """Clear all discovered types"""
        for category in self.discovered_types:
            self.discovered_types[category].clear()
    
    def print_discovered_types(self):
        """Print all discovered types in a formatted way"""
        print("\n🔍 Discovered Non-Primitive Types:")
        for category, types in self.discovered_types.items():
            if types:
                print(f"  {category.capitalize()}: {sorted(list(types))}")
            else:
                print(f"  {category.capitalize()}: None found")
    
    def extract_types_from_all_methods(self, all_methods_list):
        """
        Extract non-primitive types from a list of methods after KG traversal
        Args:
            all_methods_list: List of method dictionaries with full details
        Returns:
            Dictionary of unique non-primitive types found across all methods
        """
        unique_non_primitive_types = {
            'classes': set(),
            'interfaces': set(),
            'collections': set(),
            'generics': set(),
            'annotations': set(),
            'enums': set()
        }
        
        print(f"\n🔍 Extracting non-primitive types from {len(all_methods_list)} methods after KG traversal...")
        
        for method_detail in all_methods_list:
            if not method_detail:
                continue
                
            # Create a method_info dict in the format expected by extract_non_primitive_types
            method_for_extraction = {
                'Class': method_detail.get('Class', ''),
                'Method Name': method_detail.get('Method Name', ''),
                'Return Type': method_detail.get('Return Type', ''),
                'Parameters': method_detail.get('Parameters', ''),
                'Function Body': method_detail.get('Function Body', '')
            }
            
            # Extract types from this method
            extracted_types = self.extract_non_primitive_types(method_for_extraction)
            
            # Add to our unique collection
            for category, types in extracted_types.items():
                unique_non_primitive_types[category].update(types)
        
        # Convert sets to lists for JSON serialization
        unique_types_summary = {k: sorted(list(v)) for k, v in unique_non_primitive_types.items()}
        
        # Print summary
        print("\n🎯 UNIQUE Non-Primitive Types Found Across ALL Methods:")
        for category, types in unique_types_summary.items():
            if types:
                print(f"  {category.capitalize()}: {types}")
            else:
                print(f"  {category.capitalize()}: None found")
        
        print(f"\n📊 Total unique types found: {sum(len(v) for v in unique_types_summary.values())}")
        
        return unique_types_summary
    
    def search_top_k(self, query: str, k: int = 3):
        cleaned_query = self.clean_query_advanced(query)
        query_embedding = self.get_query_embedding(cleaned_query)
        
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            method_info = {
                'Class': self.df.iloc[idx]['Class'],
                'Method Name': self.df.iloc[idx]['Method Name'],
                'Return Type': self.df.iloc[idx]['Return Type'],
                'Parameters': self.df.iloc[idx]['Parameters']
                # Removed Function Body and type extraction from here
            }
            results.append(method_info)
        
        return results
    
def main():
    """Main function for standalone execution"""
    searcher = CodeSearcher()
    searcher.initialize()
    
    user_query = input("Enter your code search query: ")
    top_matches = searcher.search_top_k(user_query, k=3)
    
    print("\nTop 3 matching methods:")
    for i, method in enumerate(top_matches, 1):
        print(f"\n{i}. Class: {method['Class']}")
        print(f"   Method: {method['Method Name']}")
        print(f"   Return Type: {method['Return Type']}")
        print(f"   Parameters: {method['Parameters']}")
        print(f"   Method Body Preview: {method['Function Body'][:200]}...")
        print(f"   Non-Primitive Types Found: {method['Non_Primitive_Types']}")
    
    # Print all discovered types across all methods
    searcher.print_discovered_types()
    
    # Example: Get specific category of discovered types
    print(f"\n📚 All discovered classes: {searcher.get_discovered_types('classes')}")
    print(f"🗂️ All discovered collections: {searcher.get_discovered_types('collections')}")

# Usage example:
# if __name__ == "__main__":
#     main()
