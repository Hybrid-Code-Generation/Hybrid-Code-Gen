#!/usr/bin/env python3
"""
Test suite for StaticImportanceAnalyzer

This module contains comprehensive tests for the StaticImportanceAnalyzer
to ensure all components work correctly.
"""

import sys
import os
import pandas as pd
import tempfile
import json
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from static_importance_analyzer import (
    StaticImportanceAnalyzer, 
    ComplexityCalculator, 
    StaticImportanceCalculator,
    Neo4jConnection
)

def test_neo4j_connection():
    """Test Neo4j connection functionality"""
    print("🧪 Testing Neo4j Connection...")
    
    # Initialize analyzer
    analyzer = StaticImportanceAnalyzer(
        neo4j_uri="bolt://98.70.123.110:7687",
        neo4j_username="neo4j", 
        neo4j_password="y?si+:qDV3DK",
        verbose=False
    )
    
    # Test connection
    connection_result = analyzer.connect_neo4j()
    
    if connection_result['status'] == 'success':
        print("  ✅ Neo4j connection successful")
        
        # Test data extraction
        kg_data = analyzer.extract_kg_data()
        if kg_data and len(kg_data['methods']) > 0:
            print(f"  ✅ Knowledge graph data extracted: {len(kg_data['methods'])} methods, {len(kg_data['relationships'])} relationships")
        else:
            print("  ⚠️ No knowledge graph data found")
        
        # Cleanup
        analyzer.neo4j_conn.close()
        return True
    else:
        print(f"  ❌ Neo4j connection failed: {connection_result['message']}")
        return False

def test_complexity_calculator():
    """Test complexity calculation functions"""
    print("🧪 Testing Complexity Calculator...")
    
    calc = ComplexityCalculator()
    
    # Test cyclomatic complexity
    test_method = """
    public void testMethod() {
        if (condition1) {
            for (int i = 0; i < 10; i++) {
                if (condition2 && condition3) {
                    doSomething();
                } else {
                    doSomethingElse();
                }
            }
        }
        try {
            riskyOperation();
        } catch (Exception e) {
            handleError();
        }
    }
    """
    
    loc = calc.calculate_lines_of_code(test_method)
    cyclomatic = calc.calculate_cyclomatic_complexity(test_method)
    cognitive = calc.calculate_cognitive_complexity(test_method)
    halstead = calc.calculate_halstead_metrics(test_method)
    parameter_count = calc.count_parameters("String param1, int param2, boolean param3")
    parameter_complexity = calc.calculate_parameter_complexity("List<String> items, Map<String, Object> config")
    return_complexity = calc.calculate_return_type_complexity("List<Map<String, Object>>")
    
    print(f"  ✅ Lines of Code: {loc}")
    print(f"  ✅ Cyclomatic complexity: {cyclomatic}")
    print(f"  ✅ Cognitive complexity: {cognitive}")
    print(f"  ✅ Halstead effort: {halstead['effort']:.2f}")
    print(f"  ✅ Parameter count: {parameter_count}")
    print(f"  ✅ Parameter complexity: {parameter_complexity}")
    print(f"  ✅ Return type complexity: {return_complexity}")
    
    # Verify reasonable values
    assert loc > 0, "LOC should be > 0 for non-empty method"
    assert cyclomatic > 1, "Cyclomatic complexity should be > 1 for complex method"
    assert cognitive >= 0, "Cognitive complexity should be >= 0"
    assert parameter_count == 3, "Should count 3 parameters"
    assert parameter_complexity > 0, "Complex parameters should have complexity > 0"
    assert return_complexity > 0, "Complex return type should have complexity > 0"
    
    return True

def test_importance_calculator():
    """Test importance score calculation"""
    print("🧪 Testing Importance Calculator...")
    
    calc = StaticImportanceCalculator()
    
    # Create test data
    test_data = pd.DataFrame({
        'Class': ['com.example.TestClass', 'com.example.TestClass', 'com.example.AnotherClass'],
        'Method Name': ['testMethod', 'anotherMethod', 'utilMethod'],
        'LOC': [15, 8, 5],
        'Cyclomatic_Complexity': [5, 3, 1],
        'Cognitive_Complexity': [8, 4, 2],
        'Halstead_Effort': [100.5, 50.0, 20.0],
        'degree_centrality': [0.5, 0.3, 0.1],
        'betweenness_centrality': [0.3, 0.2, 0.0],
        'eigenvector_centrality': [0.4, 0.2, 0.1],
        'fan_in': [3, 1, 0],
        'fan_out': [2, 1, 0],
        'Parameter_Count': [3, 2, 1],
        'Parameter_Complexity': [2, 1, 0],
        'Return_Type_Complexity': [1, 1, 0]
    })
    
    # Calculate importance scores
    result_df = calc.calculate_importance_scores(test_data)
    
    print(f"  ✅ Processed {len(result_df)} methods")
    print(f"  ✅ Importance scores calculated")
    
    # Check required columns exist
    required_columns = ['importance_score_normalized', 'importance_category']
    for col in required_columns:
        assert col in result_df.columns, f"Missing required column: {col}"
        print(f"    ✅ {col} column present")
    
    # Verify score range
    scores = result_df['importance_score_normalized']
    assert scores.min() >= 0 and scores.max() <= 1, "Scores should be in 0-1 range"
    print(f"    ✅ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Verify categories
    categories = result_df['importance_category'].unique()
    valid_categories = ['Critical', 'High', 'Medium', 'Low', 'Minimal']
    for cat in categories:
        assert cat in valid_categories, f"Invalid category: {cat}"
    print(f"    ✅ Categories: {list(categories)}")
    
    return True

def test_data_processing():
    """Test data processing with sample AST data"""
    print("🧪 Testing Data Processing...")
    
    # Create sample AST data
    sample_data = {
        'Class': ['com.example.TestClass', 'com.example.AnotherClass', 'com.example.UtilClass'],
        'Method Name': ['mainMethod', 'processData', 'getValue'],
        'Return Type': ['void', 'List<Object>', 'String'],
        'Parameters': ['String[] args', 'List<Object> data, boolean flag', ''],
        'Function Body': [
            'public static void main(String[] args) { System.out.println("Hello"); }',
            'public void processData(List<Object> data, boolean flag) { if(flag) { for(Object obj : data) { process(obj); } } }',
            'private String getValue() { return this.value; }'
        ],
        'Package': ['com.example', 'com.example', 'com.example']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        df.to_csv(f.name, index=False)
        temp_csv_path = f.name
    
    try:
        # Initialize analyzer
        analyzer = StaticImportanceAnalyzer(
            neo4j_uri="bolt://98.70.123.110:7687",
            neo4j_username="neo4j", 
            neo4j_password="y?si+:qDV3DK",
            verbose=False
        )
        
        # Run analysis
        results = analyzer.analyze(temp_csv_path)
        
        if results is not None and len(results) > 0:
            print(f"  ✅ Processed {len(results)} methods")
            print(f"  ✅ Columns: {len(results.columns)}")
            
            # Check required columns exist
            required_columns = ['importance_score_normalized', 'importance_category']
            for col in required_columns:
                if col in results.columns:
                    print(f"    ✅ {col} column present")
                else:
                    print(f"    ❌ {col} column missing")
                    return False
            
            # Check that complexity metrics were calculated
            complexity_columns = ['LOC', 'Cyclomatic_Complexity', 'Cognitive_Complexity']
            for col in complexity_columns:
                if col in results.columns:
                    mean_val = results[col].mean()
                    print(f"    ✅ {col}: mean = {mean_val:.2f}")
                else:
                    print(f"    ❌ {col} column missing")
                    return False
            
            # Cleanup
            if analyzer.neo4j_conn:
                analyzer.neo4j_conn.close()
            
            return True
        else:
            print("  ❌ No results returned from analysis")
            return False
            
    finally:
        # Cleanup temporary file
        os.unlink(temp_csv_path)

def test_export_functionality():
    """Test CSV export functionality"""
    print("🧪 Testing Export Functionality...")
    
    # Create sample results dataframe
    sample_results = pd.DataFrame({
        'Class': ['com.example.Test'] * 3,
        'Method Name': ['method1', 'method2', 'method3'],
        'Return Type': ['void', 'String', 'int'],
        'Parameters': ['', 'String name', 'int count, boolean flag'],
        'LOC': [5, 15, 8],
        'Cyclomatic_Complexity': [1, 3, 2],
        'Cognitive_Complexity': [0, 5, 3],
        'Halstead_Effort': [10.0, 50.0, 25.0],
        'importance_score_normalized': [0.8, 0.5, 0.3],
        'importance_category': ['High', 'Medium', 'Low'],
        'fan_in': [5, 2, 0],
        'fan_out': [3, 1, 0],
        'degree_centrality': [0.4, 0.2, 0.0],
        'betweenness_centrality': [0.3, 0.1, 0.0],
        'eigenvector_centrality': [0.5, 0.2, 0.0]
    })
    
    # Initialize analyzer (no Neo4j connection needed for export test)
    analyzer = StaticImportanceAnalyzer(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="test", 
        neo4j_password="test",
        verbose=False
    )
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_output_path = f.name
    
    try:
        # Test export
        export_result = analyzer.export_results(sample_results, temp_output_path, include_all_columns=False)
        
        if export_result['status'] == 'success':
            print(f"  ✅ Export successful: {export_result['method_count']} methods, {export_result['column_count']} columns")
            print(f"  ✅ File size: {export_result['file_size_kb']} KB")
            
            # Verify file exists and has content
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                print("  ✅ Output file created and has content")
                
                # Read back and verify content
                exported_df = pd.read_csv(temp_output_path)
                print(f"  ✅ Exported file has {len(exported_df)} rows and {len(exported_df.columns)} columns")
                return True
            else:
                print("  ❌ Output file is empty or doesn't exist")
                return False
        else:
            print(f"  ❌ Export failed: {export_result['message']}")
            return False
            
    finally:
        # Cleanup
        if os.path.exists(temp_output_path):
            os.unlink(temp_output_path)

def test_integration():
    """Test complete integration with real data if available"""
    print("🧪 Testing Integration...")
    
    ast_file_path = "../AST/java_parsed.csv"
    
    if not os.path.exists(ast_file_path):
        print("  ⚠️ AST file not found, skipping integration test")
        return True
    
    # Initialize analyzer
    analyzer = StaticImportanceAnalyzer(
        neo4j_uri="bolt://98.70.123.110:7687",
        neo4j_username="neo4j", 
        neo4j_password="y?si+:qDV3DK",
        verbose=False
    )
    
    try:
        # Test connection
        connection_result = analyzer.connect_neo4j()
        if connection_result['status'] != 'success':
            print(f"  ⚠️ Neo4j connection failed, skipping integration test: {connection_result['message']}")
            return True
        
        # Run quick analysis on subset of data
        ast_df = pd.read_csv(ast_file_path)
        if len(ast_df) > 50:
            # Use subset for faster testing
            subset_path = 'temp_subset.csv'
            ast_df.head(20).to_csv(subset_path, index=False)
            
            try:
                results = analyzer.analyze(subset_path)
                if results is not None:
                    print(f"  ✅ Integration test successful: {len(results)} methods processed")
                    
                    # Test export
                    export_result = analyzer.export_results(results, 'temp_results.csv')
                    if export_result['status'] == 'success':
                        print(f"  ✅ Export test successful")
                        os.unlink('temp_results.csv')  # Cleanup
                    
                    return True
                else:
                    print("  ❌ Integration test failed: no results")
                    return False
            finally:
                if os.path.exists(subset_path):
                    os.unlink(subset_path)
        else:
            print("  ⚠️ AST file too small for subset testing")
            return True
    
    finally:
        if analyzer.neo4j_conn:
            analyzer.neo4j_conn.close()

def run_all_tests():
    """Run all tests and report results"""
    print("🚀 Starting Static Importance Analyzer Test Suite")
    print("=" * 60)
    
    tests = [
        ("Neo4j Connection", test_neo4j_connection),
        ("Complexity Calculator", test_complexity_calculator),
        ("Importance Calculator", test_importance_calculator),
        ("Data Processing", test_data_processing),
        ("Export Functionality", test_export_functionality),
        ("Integration Test", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            if test_func():
                print(f"✅ {test_name} Test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} Test FAILED")
        except Exception as e:
            print(f"❌ {test_name} Test ERROR: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The analyzer is ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
