#!/usr/bin/env python3
"""
Static Importance Analyzer for Java Methods

This script provides a comprehensive analysis system for determining the static importance
of Java methods based on various metrics including code complexity, knowledge graph
relationships, and method characteristics.

Converted from Jupyter notebook to standalone script for integration with other systems.
"""

import pandas as pd
import numpy as np
import networkx as nx
from neo4j import GraphDatabase
import json
import logging
import os
import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class Neo4jConnection:
    """Handle Neo4j database connections and queries"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    def connect(self) -> Dict[str, Any]:
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test the connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            return {"status": "success", "message": "Connected to Neo4j successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect to Neo4j: {str(e)}"}
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        if not self.driver:
            raise Exception("No active Neo4j connection")
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")

class ComplexityCalculator:
    """Calculate various complexity metrics for Java methods"""
    
    def __init__(self):
        # Java keywords that increase cyclomatic complexity
        self.decision_keywords = [
            'if', 'else', 'elif', 'while', 'for', 'switch', 'case', 
            'catch', 'try', '&&', '||', '?', 'do'
        ]
        
    def calculate_lines_of_code(self, function_body):
        """Calculate Lines of Code (LOC)"""
        if pd.isna(function_body) or function_body.strip() == "":
            return 0
        
        # Remove empty lines and comments
        lines = function_body.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('//')]
        return len(non_empty_lines)
    
    def calculate_cyclomatic_complexity(self, function_body):
        """Calculate Cyclomatic Complexity (simplified version)"""
        if pd.isna(function_body) or function_body.strip() == "":
            return 1  # Base complexity
        
        complexity = 1  # Base complexity
        
        # Count decision points
        for keyword in self.decision_keywords:
            if keyword in ['&&', '||']:
                complexity += function_body.count(keyword)
            else:
                # Use word boundaries for keywords
                pattern = r'\b' + re.escape(keyword) + r'\b'
                complexity += len(re.findall(pattern, function_body, re.IGNORECASE))
        
        return complexity
    
    def calculate_cognitive_complexity(self, function_body):
        """Calculate Cognitive Complexity (simplified)"""
        if pd.isna(function_body) or function_body.strip() == "":
            return 0
        
        cognitive = 0
        nesting_level = 0
        
        # Simple nesting and branching detection
        lines = function_body.split('\n')
        for line in lines:
            line = line.strip()
            
            # Increase nesting for blocks
            if '{' in line:
                nesting_level += line.count('{')
            if '}' in line:
                nesting_level -= line.count('}')
                nesting_level = max(0, nesting_level)
            
            # Add complexity based on constructs
            for keyword in ['if', 'while', 'for', 'switch', 'catch']:
                if keyword in line.lower():
                    cognitive += 1 + nesting_level
                    
        return cognitive
    
    def calculate_halstead_metrics(self, function_body):
        """Calculate Halstead metrics (simplified)"""
        if pd.isna(function_body) or function_body.strip() == "":
            return {'volume': 0, 'difficulty': 0, 'effort': 0}
        
        # Java operators and keywords
        operators = ['+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=', 
                    '&&', '||', '!', '++', '--', '+=', '-=', '*=', '/=']
        
        # Count unique and total operators/operands
        unique_operators = set()
        total_operators = 0
        unique_operands = set()
        total_operands = 0
        
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b|[+\-*/=<>!&|%]+', function_body)
        
        for token in tokens:
            if token in operators or any(op in token for op in operators):
                unique_operators.add(token)
                total_operators += 1
            else:
                unique_operands.add(token)
                total_operands += 1
        
        # Halstead metrics
        n1 = len(unique_operators)  # Number of distinct operators
        n2 = len(unique_operands)   # Number of distinct operands
        N1 = total_operators        # Total number of operators
        N2 = total_operands         # Total number of operands
        
        if n1 == 0 or n2 == 0:
            return {'volume': 0, 'difficulty': 0, 'effort': 0}
        
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 1 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        
        return {
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort
        }
    
    def count_parameters(self, parameters):
        """Count number of parameters"""
        if pd.isna(parameters) or parameters.strip() == "":
            return 0
        
        # Simple parameter counting
        if parameters.strip() == "":
            return 0
        
        # Split by comma and count non-empty parts
        params = [p.strip() for p in parameters.split(',') if p.strip()]
        return len(params)
    
    def calculate_parameter_complexity(self, parameters):
        """Calculate parameter type complexity"""
        if pd.isna(parameters) or parameters.strip() == "":
            return 0
        
        complexity = 0
        
        # Complex types add more complexity
        complex_types = ['List', 'Map', 'Set', 'Collection', 'Array', '[]', '<', '>']
        generic_indicators = ['<', '>', 'List', 'Map', 'Set']
        
        for complex_type in complex_types:
            complexity += parameters.count(complex_type)
        
        # Generics add extra complexity
        if any(indicator in parameters for indicator in generic_indicators):
            complexity += 2
            
        return complexity
    
    def calculate_return_type_complexity(self, return_type):
        """Calculate return type complexity"""
        if pd.isna(return_type) or return_type.strip() == "":
            return 0
        
        complexity = 1  # Base complexity for having a return type
        
        # Void methods have 0 complexity
        if return_type.lower() == 'void':
            return 0
        
        # Complex return types
        complex_indicators = ['List', 'Map', 'Set', 'Collection', '[]', '<', '>']
        for indicator in complex_indicators:
            if indicator in return_type:
                complexity += 1
        
        return complexity

class StaticImportanceCalculator:
    """Calculate static importance weights for methods"""
    
    def __init__(self):
        # Define weights for different metric categories
        self.weights = {
            # Code Complexity Metrics (35% total weight)
            'LOC': 0.07,
            'Cyclomatic_Complexity': 0.10,
            'Cognitive_Complexity': 0.08,
            'Halstead_Effort': 0.10,
            
            # Graph Centrality Metrics (30% total weight)
            'degree_centrality': 0.08,
            'betweenness_centrality': 0.08,
            'eigenvector_centrality': 0.06,
            'fan_in': 0.04,
            'fan_out': 0.04,
            
            # Parameter and Interface Metrics (20% total weight)
            'Parameter_Count': 0.07,
            'Parameter_Complexity': 0.07,
            'Return_Type_Complexity': 0.06,
            
            # Relative Importance Metrics (15% total weight)
            'class_relative_importance': 0.08,
            'name_similarity_importance': 0.07
        }
        
        # Verify weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"⚠️ Warning: Weights sum to {total_weight}, not 1.0")
            # Normalize weights
            for key in self.weights:
                self.weights[key] = self.weights[key] / total_weight
    
    def normalize_column(self, series, method='min-max'):
        """Normalize a pandas series to 0-1 range"""
        if method == 'min-max':
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - min_val) / (max_val - min_val)
        
        elif method == 'z-score':
            return (series - series.mean()) / series.std()
        
        elif method == 'robust':
            median = series.median()
            mad = (series - median).abs().median()
            if mad == 0:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - median) / (1.4826 * mad)
    
    def calculate_class_relative_importance(self, df):
        """Calculate importance relative to sibling methods in the same class"""
        class_relative_scores = []
        
        for idx, row in df.iterrows():
            class_name = row['Class']
            method_name = row['Method Name']
            
            # Get all methods in the same class
            class_methods = df[df['Class'] == class_name]
            
            if len(class_methods) <= 1:
                # Only method in class gets neutral score
                class_relative_scores.append(0.5)
                continue
            
            # Calculate relative score based on complexity within class
            method_complexity = (
                row.get('LOC', 0) * 0.3 + 
                row.get('Cyclomatic_Complexity', 0) * 0.4 + 
                row.get('Parameter_Count', 0) * 0.3
            )
            
            # Get complexity scores for all methods in class
            class_complexities = []
            for _, class_method in class_methods.iterrows():
                complexity = (
                    class_method.get('LOC', 0) * 0.3 + 
                    class_method.get('Cyclomatic_Complexity', 0) * 0.4 + 
                    class_method.get('Parameter_Count', 0) * 0.3
                )
                class_complexities.append(complexity)
            
            # Calculate relative position within class
            if max(class_complexities) == min(class_complexities):
                relative_score = 0.5
            else:
                relative_score = (method_complexity - min(class_complexities)) / (max(class_complexities) - min(class_complexities))
            
            class_relative_scores.append(relative_score)
        
        return pd.Series(class_relative_scores, index=df.index)
    
    def calculate_name_similarity_importance(self, df):
        """Calculate importance based on methods with similar names"""
        import difflib
        
        name_similarity_scores = []
        method_names = df['Method Name'].tolist()
        
        for idx, row in df.iterrows():
            method_name = row['Method Name']
            
            # Find methods with similar names (using fuzzy matching)
            similar_methods = []
            for other_name in method_names:
                if other_name != method_name:
                    similarity = difflib.SequenceMatcher(None, method_name.lower(), other_name.lower()).ratio()
                    if similarity > 0.6:  # 60% similarity threshold
                        similar_methods.append((other_name, similarity))
            
            if not similar_methods:
                # No similar methods, neutral score
                name_similarity_scores.append(0.5)
                continue
            
            # Get complexity of current method
            current_complexity = (
                row.get('LOC', 0) * 0.25 + 
                row.get('Cyclomatic_Complexity', 0) * 0.35 + 
                row.get('Parameter_Count', 0) * 0.20 +
                row.get('fan_in', 0) * 0.10 +
                row.get('fan_out', 0) * 0.10
            )
            
            # Calculate average complexity of similar methods
            similar_complexities = []
            for similar_name, _ in similar_methods:
                similar_row = df[df['Method Name'] == similar_name].iloc[0]
                similarity_complexity = (
                    similar_row.get('LOC', 0) * 0.25 + 
                    similar_row.get('Cyclomatic_Complexity', 0) * 0.35 + 
                    similar_row.get('Parameter_Count', 0) * 0.20 +
                    similar_row.get('fan_in', 0) * 0.10 +
                    similar_row.get('fan_out', 0) * 0.10
                )
                similar_complexities.append(similarity_complexity)
            
            # Calculate relative importance
            avg_similar_complexity = np.mean(similar_complexities)
            
            if avg_similar_complexity == 0:
                relative_score = 0.5
            else:
                relative_score = min(current_complexity / avg_similar_complexity, 2.0) / 2.0
            
            name_similarity_scores.append(relative_score)
        
        return pd.Series(name_similarity_scores, index=df.index)
    
    def calculate_importance_scores(self, df):
        """Calculate final importance scores with proper normalization"""
        
        # Create working copy
        working_df = df.copy()
        
        # Normalize all metrics to 0-1 scale
        metrics_to_normalize = [
            'LOC', 'Cyclomatic_Complexity', 'Cognitive_Complexity', 'Halstead_Effort',
            'degree_centrality', 'betweenness_centrality', 'eigenvector_centrality', 
            'fan_in', 'fan_out',
            'Parameter_Count', 'Parameter_Complexity', 'Return_Type_Complexity'
        ]
        
        # Normalize each metric
        for metric in metrics_to_normalize:
            if metric in working_df.columns:
                normalized_col = f"{metric}_normalized"
                working_df[normalized_col] = self.normalize_column(working_df[metric], method='min-max')
        
        # Calculate relative importance metrics
        working_df['class_relative_importance'] = self.calculate_class_relative_importance(working_df)
        working_df['name_similarity_importance'] = self.calculate_name_similarity_importance(working_df)
        
        # Calculate weighted importance score
        importance_scores = []
        
        for idx, row in working_df.iterrows():
            score = 0.0
            
            # Sum weighted normalized metrics
            for metric, weight in self.weights.items():
                if metric in ['class_relative_importance', 'name_similarity_importance']:
                    # These are already normalized
                    value = row.get(metric, 0.5)
                else:
                    # Use normalized version
                    normalized_col = f"{metric}_normalized"
                    value = row.get(normalized_col, 0.0)
                
                score += value * weight
            
            importance_scores.append(score)
        
        # Add raw importance scores
        working_df['importance_score_raw'] = importance_scores
        
        # Normalize final scores to 0-1 range
        working_df['importance_score_normalized'] = self.normalize_column(
            working_df['importance_score_raw'], method='min-max'
        )
        
        # Categorize importance levels
        def categorize_importance(score):
            if score >= 0.9:
                return "Critical"
            elif score >= 0.75:
                return "High"
            elif score >= 0.5:
                return "Medium"
            elif score >= 0.25:
                return "Low"
            else:
                return "Minimal"
        
        working_df['importance_category'] = working_df['importance_score_normalized'].apply(categorize_importance)
        
        return working_df

class StaticImportanceAnalyzer:
    """Main analyzer class that orchestrates the entire analysis process"""
    
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str, verbose: bool = False):
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.verbose = verbose
        
        # Initialize components
        self.neo4j_conn = None
        self.complexity_calc = ComplexityCalculator()
        self.importance_calc = StaticImportanceCalculator()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def connect_neo4j(self) -> Dict[str, Any]:
        """Establish Neo4j connection"""
        self.neo4j_conn = Neo4jConnection(self.neo4j_uri, self.neo4j_username, self.neo4j_password)
        result = self.neo4j_conn.connect()
        
        if result['status'] == 'success' and self.verbose:
            self.logger.info("✅ Connected to Neo4j successfully")
        elif result['status'] == 'error':
            self.logger.error(f"❌ Neo4j connection failed: {result['message']}")
        
        return result
    
    def extract_kg_data(self) -> Optional[Dict[str, Any]]:
        """Extract knowledge graph data from Neo4j"""
        if not self.neo4j_conn:
            connection_result = self.connect_neo4j()
            if connection_result['status'] != 'success':
                return None
        
        try:
            # Get all method nodes
            methods_query = """
            MATCH (m:Method)
            RETURN m.name as method_name, 
                   id(m) as node_id,
                   m.depth as depth,
                   labels(m) as labels
            """
            
            # Get CALLS and CALLED_BY relationships specifically
            calls_relationships_query = """
            MATCH (m1:Method)-[r:CALLS]->(m2:Method)
            RETURN m1.name as source_method,
                   m2.name as target_method,
                   'CALLS' as relationship_type,
                   id(m1) as source_id,
                   id(m2) as target_id
            UNION ALL
            MATCH (m1:Method)-[r:CALLED_BY]->(m2:Method)
            RETURN m1.name as source_method,
                   m2.name as target_method,
                   'CALLED_BY' as relationship_type,
                   id(m1) as source_id,
                   id(m2) as target_id
            """
            
            if self.verbose:
                self.logger.info("🔍 Extracting methods from knowledge graph...")
            methods_data = self.neo4j_conn.query(methods_query)
            methods_df = pd.DataFrame([dict(record) for record in methods_data])
            
            if self.verbose:
                self.logger.info("🔍 Extracting CALLS and CALLED_BY relationships...")
            relationships_data = self.neo4j_conn.query(calls_relationships_query)
            relationships_df = pd.DataFrame([dict(record) for record in relationships_data])
            
            if self.verbose:
                self.logger.info(f"📊 Found {len(methods_df)} methods and {len(relationships_df)} relationships")
            
            return {
                'methods': methods_df,
                'relationships': relationships_df
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract KG data: {str(e)}")
            return None
    
    def compute_centrality_measures(self, kg_data: Dict[str, Any], enhanced_df: pd.DataFrame) -> pd.DataFrame:
        """Compute centrality measures using CALLS and CALLED_BY relationships"""
        
        if not kg_data or kg_data['relationships'].empty:
            if self.verbose:
                self.logger.warning("⚠️ No relationship data available for centrality calculation")
            # Return default values
            return pd.DataFrame({
                'method_name': enhanced_df['Method Name'],
                'degree_centrality': 0.0,
                'betweenness_centrality': 0.0,
                'eigenvector_centrality': 0.0,
                'fan_in': 0,
                'fan_out': 0
            })
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all methods as nodes
        all_methods = set(enhanced_df['Method Name'].tolist())
        if not kg_data['methods'].empty:
            all_methods.update(kg_data['methods']['method_name'].tolist())
        
        G.add_nodes_from(all_methods)
        
        # Process relationships
        relationships_df = kg_data['relationships']
        if self.verbose:
            self.logger.info(f"🔄 Processing {len(relationships_df)} relationships...")
        
        # Track edges to avoid duplicates
        edges_added = 0
        
        for _, row in relationships_df.iterrows():
            source = str(row['source_method'])
            target = str(row['target_method'])
            rel_type = str(row['relationship_type'])
            
            if source and target and source != target:
                if rel_type == 'CALLS':
                    # source calls target
                    G.add_edge(source, target)
                    edges_added += 1
                elif rel_type == 'CALLED_BY':
                    # source is called by target, so target calls source
                    G.add_edge(target, source)
                    edges_added += 1
        
        if self.verbose:
            self.logger.info(f"📊 Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Calculate metrics
        fan_in = {node: G.in_degree(node) for node in G.nodes()}
        fan_out = {node: G.out_degree(node) for node in G.nodes()}
        
        # Calculate centralities
        degree_cent = nx.degree_centrality(G)
        
        try:
            between_cent = nx.betweenness_centrality(G, k=min(50, G.number_of_nodes()))
        except:
            between_cent = {node: 0.0 for node in G.nodes()}
        
        try:
            eigen_cent = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        except:
            eigen_cent = degree_cent.copy()
        
        # Create result dataframe
        results = []
        for method in all_methods:
            results.append({
                'method_name': method,
                'degree_centrality': degree_cent.get(method, 0.0),
                'betweenness_centrality': between_cent.get(method, 0.0),
                'eigenvector_centrality': eigen_cent.get(method, 0.0),
                'fan_in': fan_in.get(method, 0),
                'fan_out': fan_out.get(method, 0)
            })
        
        return pd.DataFrame(results)
    
    def analyze(self, ast_csv_path: str) -> Optional[pd.DataFrame]:
        """Main analysis function that processes AST data and calculates importance scores"""
        
        try:
            # Step 1: Load AST data
            if self.verbose:
                self.logger.info(f"📂 Loading AST data from {ast_csv_path}")
            
            if not os.path.exists(ast_csv_path):
                self.logger.error(f"❌ AST file not found: {ast_csv_path}")
                return None
            
            df = pd.read_csv(ast_csv_path)
            
            if self.verbose:
                self.logger.info(f"📊 Loaded {len(df)} methods from AST data")
            
            # Step 2: Extract knowledge graph data
            if self.verbose:
                self.logger.info("🔍 Extracting knowledge graph data...")
            
            kg_data = self.extract_kg_data()
            if not kg_data:
                self.logger.error("❌ Failed to extract knowledge graph data")
                return None
            
            # Step 3: Calculate complexity metrics
            if self.verbose:
                self.logger.info("⚙️ Calculating complexity metrics...")
            
            df['LOC'] = df['Function Body'].apply(self.complexity_calc.calculate_lines_of_code)
            df['Cyclomatic_Complexity'] = df['Function Body'].apply(self.complexity_calc.calculate_cyclomatic_complexity)
            df['Cognitive_Complexity'] = df['Function Body'].apply(self.complexity_calc.calculate_cognitive_complexity)
            
            # Calculate Halstead metrics
            halstead_metrics = df['Function Body'].apply(self.complexity_calc.calculate_halstead_metrics)
            df['Halstead_Volume'] = [h['volume'] for h in halstead_metrics]
            df['Halstead_Difficulty'] = [h['difficulty'] for h in halstead_metrics]
            df['Halstead_Effort'] = [h['effort'] for h in halstead_metrics]
            
            # Parameter analysis
            df['Parameter_Count'] = df['Parameters'].apply(self.complexity_calc.count_parameters)
            df['Parameter_Complexity'] = df['Parameters'].apply(self.complexity_calc.calculate_parameter_complexity)
            df['Return_Type_Complexity'] = df['Return Type'].apply(self.complexity_calc.calculate_return_type_complexity)
            
            # Step 4: Calculate graph metrics
            if self.verbose:
                self.logger.info("📈 Calculating graph centrality measures...")
            
            centrality_df = self.compute_centrality_measures(kg_data, df)
            
            # Step 5: Merge centrality data with enhanced dataframe
            df = df.merge(centrality_df, left_on='Method Name', right_on='method_name', how='left')
            
            # Fill missing centrality values with 0
            centrality_columns = ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'fan_in', 'fan_out']
            for col in centrality_columns:
                df[col] = df[col].fillna(0)
            
            # Step 6: Calculate importance scores
            if self.verbose:
                self.logger.info("⭐ Calculating importance scores...")
            
            final_df = self.importance_calc.calculate_importance_scores(df)
            
            # Step 7: Sort by importance score
            final_df = final_df.sort_values('importance_score_normalized', ascending=False).reset_index(drop=True)
            
            if self.verbose:
                self.logger.info("✅ Analysis completed successfully!")
                self.logger.info(f"📊 Total methods analyzed: {len(final_df)}")
                category_counts = final_df['importance_category'].value_counts()
                for category, count in category_counts.items():
                    self.logger.info(f"  • {category}: {count} methods")
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def export_results(self, df: pd.DataFrame, output_path: str, include_all_columns: bool = False) -> Dict[str, Any]:
        """Export analysis results to CSV with optional column filtering"""
        
        try:
            if include_all_columns:
                # Export all columns
                export_df = df.copy()
            else:
                # Export essential columns only
                essential_columns = [
                    'Class', 'Method Name', 'Return Type', 'Parameters',
                    'LOC', 'Cyclomatic_Complexity', 'Cognitive_Complexity', 'Halstead_Effort',
                    'degree_centrality', 'betweenness_centrality', 'eigenvector_centrality',
                    'fan_in', 'fan_out', 'Parameter_Count', 'Parameter_Complexity', 'Return_Type_Complexity',
                    'importance_score_normalized', 'importance_category'
                ]
                
                # Only include columns that exist in the dataframe
                available_columns = [col for col in essential_columns if col in df.columns]
                export_df = df[available_columns].copy()
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Export to CSV
            export_df.to_csv(output_path, index=False)
            
            # Calculate file statistics
            file_size_kb = round(os.path.getsize(output_path) / 1024, 2)
            
            # Calculate correlation between fan_in and fan_out for validation
            correlation = df[['fan_in', 'fan_out']].corr().iloc[0, 1] if len(df) > 1 else 0
            
            if self.verbose:
                self.logger.info(f"📁 Results exported to: {output_path}")
                self.logger.info(f"📈 File contains {len(export_df)} methods and {len(export_df.columns)} columns")
                self.logger.info(f"💾 File size: {file_size_kb} KB")
                self.logger.info(f"🔗 Fan-in/Fan-out correlation: {correlation:.3f}")
            
            return {
                'status': 'success',
                'file_path': output_path,
                'method_count': len(export_df),
                'column_count': len(export_df.columns),
                'file_size_kb': file_size_kb,
                'fan_in_fan_out_correlation': round(correlation, 3)
            }
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'status': 'error',
                'message': error_msg
            }
    
    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        
        try:
            summary = {
                'dataset_overview': {
                    'total_methods': len(df),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'unique_classes': df['Class'].nunique() if 'Class' in df.columns else 0
                },
                'importance_distribution': {},
                'graph_metrics': {},
                'complexity_metrics': {},
                'top_methods': []
            }
            
            # Importance distribution
            if 'importance_category' in df.columns:
                category_counts = df['importance_category'].value_counts()
                for category, count in category_counts.items():
                    percentage = round((count / len(df)) * 100, 1)
                    summary['importance_distribution'][category] = {
                        'count': int(count),
                        'percentage': percentage
                    }
            
            # Graph metrics summary
            if 'fan_in' in df.columns and 'fan_out' in df.columns:
                summary['graph_metrics'] = {
                    'fan_in': {
                        'mean': round(df['fan_in'].mean(), 2),
                        'max': int(df['fan_in'].max()),
                        'methods_with_nonzero': int((df['fan_in'] > 0).sum())
                    },
                    'fan_out': {
                        'mean': round(df['fan_out'].mean(), 2),
                        'max': int(df['fan_out'].max()),
                        'methods_with_nonzero': int((df['fan_out'] > 0).sum())
                    }
                }
            
            # Complexity metrics summary
            complexity_columns = ['LOC', 'Cyclomatic_Complexity', 'Cognitive_Complexity', 'Halstead_Effort']
            for col in complexity_columns:
                if col in df.columns:
                    summary['complexity_metrics'][col] = {
                        'mean': round(df[col].mean(), 2),
                        'max': int(df[col].max()),
                        'std': round(df[col].std(), 2)
                    }
            
            # Top 5 methods by importance
            score_column = 'importance_score_normalized' if 'importance_score_normalized' in df.columns else 'importance_score_raw'
            if score_column in df.columns:
                top_methods = df.head(5)
                for _, method in top_methods.iterrows():
                    summary['top_methods'].append({
                        'class': method.get('Class', 'N/A'),
                        'method': method.get('Method Name', 'N/A'),
                        'score': round(method.get(score_column, 0), 4),
                        'category': method.get('importance_category', 'N/A')
                    })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Summary generation failed: {str(e)}")
            return {'error': str(e)}
    
    def __del__(self):
        """Cleanup Neo4j connection on object destruction"""
        if self.neo4j_conn:
            self.neo4j_conn.close()

def main():
    """Main function for standalone execution"""
    
    # Configuration - Update these with your actual values
    config = {
        'neo4j_uri': "bolt://98.70.123.110:7687",
        'neo4j_username': "neo4j", 
        'neo4j_password': "y?si+:qDV3DK",
        'ast_csv_path': "../AST/java_parsed.csv",
        'output_csv_path': f"java_methods_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    }
    
    print("🚀 Starting Java Method Static Importance Analysis")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = StaticImportanceAnalyzer(
        neo4j_uri=config['neo4j_uri'],
        neo4j_username=config['neo4j_username'],
        neo4j_password=config['neo4j_password'],
        verbose=True
    )
    
    try:
        # Run the complete analysis
        results_df = analyzer.analyze(config['ast_csv_path'])
        
        if results_df is not None:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📊 Analyzed {len(results_df)} methods")
            
            # Export results to CSV
            export_stats = analyzer.export_results(
                results_df, 
                config['output_csv_path'],
                include_all_columns=False
            )
            
            if export_stats['status'] == 'success':
                print(f"\n📁 Results exported to: {export_stats['file_path']}")
                print(f"📈 File contains {export_stats['method_count']} methods and {export_stats['column_count']} columns")
                print(f"💾 File size: {export_stats['file_size_kb']} KB")
                print(f"🔗 Fan-in/Fan-out correlation: {export_stats['fan_in_fan_out_correlation']}")
                
                # Generate comprehensive summary
                summary = analyzer.generate_summary(results_df)
                
                # Save summary to JSON
                summary_path = config['output_csv_path'].replace('.csv', '_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"📋 Summary saved to: {summary_path}")
                
                # Display key insights
                print("\n🎯 KEY INSIGHTS")
                print("-" * 30)
                
                if 'importance_distribution' in summary:
                    print(f"⭐ Importance Distribution:")
                    for category, info in summary['importance_distribution'].items():
                        print(f"  • {category}: {info['count']} methods ({info['percentage']}%)")
                
                if 'top_methods' in summary:
                    print(f"\n🏆 Top 5 Most Important Methods:")
                    for i, method in enumerate(summary['top_methods'], 1):
                        category = method.get('category', 'N/A')
                        print(f"  {i}. {method['class']}.{method['method']} (score: {method['score']}, {category})")
                
                print(f"\n🎉 Analysis complete! Ready for hybrid RAG system integration.")
                
            else:
                print(f"❌ Export failed: {export_stats['message']}")
        else:
            print("❌ Analysis failed - check Neo4j connection and AST file path")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
