from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import json

def search_method(
    method_name: str,
    class_name: str,
    return_type: str,
    parameters: str,
    calls: int = 3,
    called_by: int = 1,
    belongs_to: bool = True,
    uses: bool = True
):
    if not method_name:
        print("Error: Method name must be provided.")
        raise ValueError("Method name must be provided.")
    
    # Setup connection details
    NEO4J_URI = "bolt://172.203.167.64:7687"  # Update if using remote DB
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "C{&K1r.eZ9*4"

    # Establish a connection
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def create_method_hash(method_name: str, class_name: str, parameters: str, return_type: str) -> str:
        """Create a unique hash for a method based on its properties"""
        # Combine all properties to create a unique identifier
        unique_string = f"{method_name}|{class_name or ''}|{parameters or ''}|{return_type or ''}"
        return unique_string

    def parse_method_hash(method_hash: str) -> Dict[str, str]:
        """Parse a method hash back into its components"""
        parts = method_hash.split("|")
        return {
            "method_name": parts[0] if len(parts) > 0 else "",
            "class_name": parts[1] if len(parts) > 1 else "",
            "parameters": parts[2] if len(parts) > 2 else "",
            "return_type": parts[3] if len(parts) > 3 else ""
        }

    def fetch_all_calls_map(tx) -> Dict[str, List[str]]:
        query = """
        MATCH p=(caller:Method)-[r:CALLS]->(callee:Method) RETURN caller.name AS callerMethodName, caller.className as callerClassName, caller.parameters as callerParameters, caller.returnType as callerReturnType, callee.name AS calleeMethodName, callee.returnType as calleeReturnType, callee.parameters as calleeParameters, callee.className as calleeClassName;
        """
        result = tx.run(query)
        calls_map = {}
        for r in result:
            caller_hash = create_method_hash(
                r["callerMethodName"],
                r["callerClassName"],
                r["callerParameters"],
                r["callerReturnType"]
            )
            callee_hash = create_method_hash(
                r["calleeMethodName"],
                r["calleeClassName"],
                r["calleeParameters"],
                r["calleeReturnType"]
            )
            calls_map.setdefault(caller_hash, []).append(callee_hash)
        return calls_map

    def fetch_all_called_by_map(tx) -> Dict[str, List[str]]:
        query = """
        MATCH p=(caller:Method)-[r:CALLED_BY]->(callee:Method) RETURN caller.name AS callerMethodName, caller.className as callerClassName, caller.parameters as callerParameters, caller.returnType as callerReturnType, callee.name AS calleeMethodName, callee.returnType as calleeReturnType, callee.parameters as calleeParameters, callee.className as calleeClassName;
        """
        result = tx.run(query)
        calls_map = {}
        for r in result:
            caller_hash = create_method_hash(
                r["callerMethodName"],
                r["callerClassName"],
                r["callerParameters"],
                r["callerReturnType"]
            )
            callee_hash = create_method_hash(
                r["calleeMethodName"],
                r["calleeClassName"],
                r["calleeParameters"],
                r["calleeReturnType"]
            )
            calls_map.setdefault(caller_hash, []).append(callee_hash)
        return calls_map

    def dfs_calls(method_hash: str, calls_map: Dict[str, List[str]], max_depth: int) -> List[Dict[str, Any]]:
        visited = set()
        result = []

        def dfs(node_hash: str, level: int, parent_hash: Optional[str]):
            if level > max_depth or node_hash in visited:
                return
            visited.add(node_hash)
            
            # Parse the hash to get method details
            method_details = parse_method_hash(node_hash)
            parent_details = parse_method_hash(parent_hash) if parent_hash else None
            
            # Skip entries with empty or None class_name
            if method_details["class_name"] and method_details["class_name"] != "":
                result.append({
                    "method_name": method_details["method_name"],
                    "class_name": method_details["class_name"],
                    "parameters": method_details["parameters"],
                    "return_type": method_details["return_type"],
                    "depth": level,
                    "parent": parent_details["method_name"] if parent_details else None,
                    "level": level
                })
            
            for child_hash in calls_map.get(node_hash, []):
                dfs(child_hash, level + 1, node_hash)

        dfs(method_hash, level=1, parent_hash=None)
        return result

    def fetch_calls_python(tx, method_name: str, class_name: str, parameters: str, return_type: str, depth: int) -> List[Dict[str, Any]]:
        calls_map = fetch_all_calls_map(tx)
        key = create_method_hash(method_name, class_name, parameters, return_type)
        return dfs_calls(key, calls_map, depth)

    def fetch_called_by_python(tx, method_name: str, class_name: str, parameters: str, return_type: str, depth: int) -> List[Dict[str, Any]]:
        called_by_map = fetch_all_called_by_map(tx)
        key = create_method_hash(method_name, class_name, parameters, return_type)
        return dfs_calls(key, called_by_map, depth)

    def fetch_belongs_to(tx, method_name: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (m:Method {name: $method_name})-[:BELONGS_TO]->(c:Class)
        RETURN c.name AS class_name, c.depth AS depth
        """
        record = tx.run(query, method_name=method_name).single()
        if record:
            return {
                "class_name": record["class_name"],
                "depth": record["depth"] or 0,
                "parent": method_name,
                "level": 1
            }
        return None

    def fetch_uses(tx, method_name: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (m:Method {name: $method_name})-[:USES]->(v:Variable)
        RETURN v.name AS variable_name, v.depth AS depth
        """
        result = tx.run(query, method_name=method_name)
        return [
            {
                "variable_name": r["variable_name"],
                "depth": r["depth"] or 0,
                "parent": method_name,
                "level": 1
            } for r in result
        ]

    def retrieve_kg_context(
        method_name: str,
        class_name: str,
        parameters: str,
        return_type: str,
        calls: int = 0,
        called_by: int = 0,
        belongs_to: bool = False,
        uses: bool = False
    ) -> Dict[str, Any]:
        result = {
            "method_name": method_name,
            "depth": max(calls, called_by),
            "level": 0,
            "CALLS": [],
            "CALLED_BY": [],
            "BELONGS_TO": {},
            "USES": [],
        }

        with driver.session() as session:
            if calls > 0:
                result["CALLS"] = session.execute_read(fetch_calls_python, method_name,
                class_name, parameters, return_type, calls)
            if called_by > 0:
                result["CALLED_BY"] = session.execute_read(fetch_called_by_python, method_name,
                class_name, parameters, return_type, called_by)
            if belongs_to:
                belongs = session.execute_read(fetch_belongs_to, method_name)
                if belongs:
                    result["BELONGS_TO"] = belongs
            if uses:
                result["USES"] = session.execute_read(fetch_uses, method_name)

        return result

    kg_data = retrieve_kg_context(
        method_name=method_name,
        class_name=class_name,
        parameters=parameters,
        return_type=return_type,
        calls=calls,
        called_by=called_by,
        belongs_to=belongs_to,
        uses=uses
    )

    # Pretty print
    # print(json.dumps(kg_data, indent=2))

    return kg_data

    # Close the driver connection
    driver.close()

# Usage example:
# if __name__ == "__main__":
#     search_method(method_name="getPreferences", calls=3, called_by=1, belongs_to=True, uses=True)
