#!/usr/bin/env python3
"""
Example usage of the StaticImportanceAnalyzer

This script demonstrates how to use the StaticImportanceAnalyzer class
to analyze Java methods and generate importance scores for a hybrid RAG system.
"""

from static_importance_analyzer import StaticImportanceAnalyzer
import json
import os

def analyze_java_methods():
    """Example function showing how to use the analyzer"""
    
    # Configuration - Update these with your actual values
    config = {
        'neo4j_uri': "bolt://98.70.123.110:7687",
        'neo4j_username': "neo4j", 
        'neo4j_password': "y?si+:qDV3DK",
        'ast_csv_path': "../AST/java_parsed.csv",
        'output_csv_path': "java_methods_importance_results.csv"
    }
    
    print("🚀 Starting Java Method Importance Analysis")
    print("=" * 50)
    
    # Initialize the analyzer
    analyzer = StaticImportanceAnalyzer(
        neo4j_uri=config['neo4j_uri'],
        neo4j_username=config['neo4j_username'],
        neo4j_password=config['neo4j_password'],
        verbose=True  # Set to False for less output
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
                include_all_columns=False  # Set to True to include all intermediate columns
            )
            
            if export_stats['status'] == 'success':
                print(f"\n📁 Results exported to: {export_stats['file_path']}")
                print(f"📈 File contains {export_stats['method_count']} methods and {export_stats['column_count']} columns")
                print(f"💾 File size: {export_stats['file_size_kb']} KB")
                
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
                
                if 'graph_metrics' in summary and 'fan_in' in summary['graph_metrics']:
                    fan_in_methods = summary['graph_metrics']['fan_in']['methods_with_nonzero']
                    fan_out_methods = summary['graph_metrics']['fan_out']['methods_with_nonzero']
                    total_methods = summary['dataset_overview']['total_methods']
                    
                    print(f"🕸️ Graph Connectivity:")
                    print(f"  • {fan_in_methods}/{total_methods} methods have incoming calls")
                    print(f"  • {fan_out_methods}/{total_methods} methods make outgoing calls")
                    
                    if 'fan_in_fan_out_correlation' in export_stats:
                        correlation = export_stats['fan_in_fan_out_correlation']
                        print(f"  • Fan-in/Fan-out correlation: {correlation}")
                
                if 'importance_distribution' in summary:
                    print(f"\n⭐ Importance Distribution:")
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

def analyze_with_custom_config():
    """Example showing how to use custom configuration"""
    
    # Custom configuration example
    custom_config = {
        'neo4j_uri': "bolt://localhost:7687",  # Your Neo4j instance
        'neo4j_username': "neo4j",
        'neo4j_password': "your_password",
        'ast_csv_path': "path/to/your/ast_data.csv"
    }
    
    # Initialize with custom settings
    analyzer = StaticImportanceAnalyzer(
        neo4j_uri=custom_config['neo4j_uri'],
        neo4j_username=custom_config['neo4j_username'],
        neo4j_password=custom_config['neo4j_password'],
        verbose=False  # Minimal output
    )
    
    # Test connection first
    connection_info = analyzer.connect_neo4j()
    if connection_info['status'] == 'success':
        print("✅ Neo4j connection successful")
        
        # Run analysis
        results = analyzer.analyze(custom_config['ast_csv_path'])
        
        if results is not None:
            # Custom export with all columns
            analyzer.export_results(
                results, 
                "full_analysis_results.csv",
                include_all_columns=True
            )
            print("✅ Full analysis exported")
        
        # Close connection
        if analyzer.neo4j_conn:
            analyzer.neo4j_conn.close()
    else:
        print(f"❌ Connection failed: {connection_info['message']}")

def quick_analysis():
    """Quick analysis with minimal configuration"""
    
    # Minimal setup for testing
    analyzer = StaticImportanceAnalyzer(
        neo4j_uri="bolt://98.70.123.110:7687",
        neo4j_username="neo4j", 
        neo4j_password="y?si+:qDV3DK",
        verbose=True
    )
    
    # Test just the Neo4j connection
    connection_info = analyzer.connect_neo4j()
    print(f"Connection status: {connection_info['status']}")
    
    if connection_info['status'] == 'success':
        # Extract just the knowledge graph data
        kg_data = analyzer.extract_kg_data()
        if kg_data:
            print(f"Found {len(kg_data['methods'])} methods in knowledge graph")
            print(f"Found {len(kg_data['relationships'])} relationships")
    
    # Clean up
    if analyzer.neo4j_conn:
        analyzer.neo4j_conn.close()

if __name__ == "__main__":
    # Run the main analysis
    analyze_java_methods()
    
    # Uncomment to run other examples:
    # analyze_with_custom_config()
    # quick_analysis()
