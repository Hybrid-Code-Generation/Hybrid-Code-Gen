# Static Importance Analyzer - Script Version

A comprehensive Python script for analyzing the static importance of Java methods based on code complexity, knowledge graph relationships, and method characteristics. This script is converted from the Jupyter notebook for easy integration with other systems.

## Features

- **Multi-metric Analysis**: Combines complexity metrics, graph centrality, fan-in/out analysis, and method characteristics
- **Neo4j Integration**: Extracts knowledge graph data from Neo4j databases with CALLS and CALLED_BY relationships
- **NetworkX Graph Analysis**: Calculates centrality metrics (betweenness, closeness, eigenvector)
- **Comprehensive Complexity Metrics**: LOC, Cyclomatic, Cognitive, and Halstead complexity measures
- **Flexible Export**: Supports both essential and comprehensive CSV exports
- **Comprehensive Testing**: Includes full test suite for validation
- **Detailed Logging**: Verbose mode for debugging and progress tracking

## Installation

### Prerequisites

```bash
pip install pandas numpy networkx neo4j
```

### Dependencies

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- networkx >= 2.8
- neo4j >= 5.0.0

## Quick Start

```python
from static_importance_analyzer import StaticImportanceAnalyzer

# Initialize the analyzer
analyzer = StaticImportanceAnalyzer(
    neo4j_uri="bolt://your-neo4j-server:7687",
    neo4j_username="neo4j",
    neo4j_password="your_password",
    verbose=True
)

# Run analysis
results_df = analyzer.analyze("path/to/your/ast_data.csv")

# Export results
analyzer.export_results(results_df, "importance_results.csv")
```

## Architecture

### Core Classes

#### `StaticImportanceAnalyzer`
Main orchestrator class that coordinates the entire analysis pipeline.

**Key Methods:**
- `analyze(ast_csv_path)`: Complete analysis pipeline
- `export_results(df, output_path)`: Export results to CSV
- `generate_summary(df)`: Generate analysis summary

#### `Neo4jConnection`
Handles Neo4j database connections and query execution.

**Key Methods:**
- `connect()`: Establish database connection
- `query(query, parameters)`: Run Cypher queries

#### `ComplexityCalculator`
Calculates various code complexity metrics.

**Metrics:**
- Lines of Code (LOC)
- Cyclomatic complexity (control flow analysis)
- Cognitive complexity (mental complexity)
- Halstead metrics (volume, difficulty, effort)
- Parameter analysis (count and complexity)
- Return type complexity

#### `StaticImportanceCalculator`
Computes weighted importance scores from multiple metrics.

**Score Components:**
- Code Complexity (35% weight): LOC, Cyclomatic, Cognitive, Halstead
- Graph Centrality (30% weight): Degree, Betweenness, Eigenvector, Fan-in/out
- Parameter & Interface (20% weight): Parameter count/complexity, Return type
- Relative Importance (15% weight): Class-relative importance, Name similarity

## Usage Examples

### Basic Analysis

```python
from static_importance_analyzer import StaticImportanceAnalyzer

# Configuration
config = {
    'neo4j_uri': "bolt://98.70.123.110:7687",
    'neo4j_username': "neo4j",
    'neo4j_password': "your_password",
    'ast_csv_path': "java_parsed.csv"
}

# Initialize and run
analyzer = StaticImportanceAnalyzer(
    neo4j_uri=config['neo4j_uri'],
    neo4j_username=config['neo4j_username'],
    neo4j_password=config['neo4j_password'],
    verbose=True
)

results = analyzer.analyze(config['ast_csv_path'])
```

### Custom Export Options

```python
# Export essential columns only (default)
analyzer.export_results(results, "essential_results.csv", include_all_columns=False)

# Export all columns including intermediate calculations
analyzer.export_results(results, "full_results.csv", include_all_columns=True)
```

### Connection Testing

```python
# Test Neo4j connection before analysis
connection_info = analyzer.connect_neo4j()
if connection_info['status'] == 'success':
    print("✅ Connected successfully")
else:
    print(f"❌ Connection failed: {connection_info['message']}")
```

## Input Data Format

### AST CSV Requirements

The AST CSV file should contain at minimum:

| Column | Description | Example |
|--------|-------------|---------|
| `Class` | Full class name | `com.example.MyClass` |
| `Method Name` | Method name | `processData` |
| `Return Type` | Method return type | `List<String>` |
| `Parameters` | Method parameters | `String name, int count` |
| `Function Body` | Method body code | `public void method() { ... }` |
| `Package` | Package name | `com.example` |

### Neo4j Knowledge Graph

Expected relationship types:
- `CALLS`: Method A calls Method B
- `CALLED_BY`: Method A is called by Method B

Node properties:
- `name`: Method name
- Additional properties as available

## Output Format

### Essential Columns Export

| Column | Description |
|--------|-------------|
| `Class` | Class name |
| `Method Name` | Method name |
| `Return Type` | Method return type |
| `Parameters` | Method parameters |
| `LOC` | Lines of code |
| `Cyclomatic_Complexity` | Control flow complexity |
| `Cognitive_Complexity` | Mental complexity |
| `Halstead_Effort` | Halstead effort metric |
| `degree_centrality` | Graph degree centrality |
| `betweenness_centrality` | Graph betweenness centrality |
| `eigenvector_centrality` | Graph eigenvector centrality |
| `fan_in` | Number of incoming calls |
| `fan_out` | Number of outgoing calls |
| `Parameter_Count` | Number of parameters |
| `Parameter_Complexity` | Parameter type complexity |
| `Return_Type_Complexity` | Return type complexity |
| `importance_score_normalized` | Final importance score (0-1) |
| `importance_category` | Category: Critical/High/Medium/Low/Minimal |

### Summary JSON Output

```json
{
  "dataset_overview": {
    "total_methods": 1500,
    "analysis_timestamp": "2024-01-15T10:30:00",
    "unique_classes": 45
  },
  "importance_distribution": {
    "Critical": {"count": 75, "percentage": 5.0},
    "High": {"count": 225, "percentage": 15.0},
    "Medium": {"count": 375, "percentage": 25.0},
    "Low": {"count": 375, "percentage": 25.0},
    "Minimal": {"count": 450, "percentage": 30.0}
  },
  "graph_metrics": {
    "fan_in": {"mean": 2.3, "max": 15, "methods_with_nonzero": 800},
    "fan_out": {"mean": 1.8, "max": 12, "methods_with_nonzero": 750}
  },
  "top_methods": [
    {
      "class": "com.example.Core",
      "method": "processMain",
      "score": 0.9245,
      "category": "Critical"
    }
  ]
}
```

## Testing

Run the comprehensive test suite:

```bash
python test_analyzer.py
```

Test coverage includes:
- Neo4j connectivity
- Complexity calculations
- Importance scoring
- Data processing pipeline
- Export functionality
- Integration testing

## File Structure

```
static_importance_analysis/
├── static_importance_analyzer.py    # Main analyzer script
├── example_usage.py                 # Usage examples
├── test_analyzer.py                 # Test suite
└── README.md                        # This file
```

## Configuration

### Neo4j Setup

Ensure your Neo4j database contains:
1. Method nodes with required properties
2. CALLS and CALLED_BY relationships
3. Proper indexing for performance

### Performance Optimization

For large datasets:
- Use Neo4j indexes on method properties
- Consider batch processing for very large AST files
- Use `verbose=False` for production runs

## Methodology

### Importance Score Calculation

The importance score is calculated using a weighted combination of normalized metrics:

1. **Code Complexity (35%)**: LOC (7%), Cyclomatic (10%), Cognitive (8%), Halstead (10%)
2. **Graph Centrality (30%)**: Degree (8%), Betweenness (8%), Eigenvector (6%), Fan-in (4%), Fan-out (4%)
3. **Interface Complexity (20%)**: Parameter count (7%), Parameter complexity (7%), Return type (6%)
4. **Relative Importance (15%)**: Class-relative (8%), Name similarity (7%)

### Normalization Process

1. **Individual Metric Normalization**: Each metric is normalized to 0-1 range using min-max scaling
2. **Weighted Combination**: Metrics are combined using predefined weights
3. **Final Normalization**: Combined scores are normalized again for final 0-1 range
4. **Categorization**: Methods are categorized into 5 levels based on percentile thresholds

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Check server URL and credentials
   - Verify network connectivity
   - Ensure Neo4j is running

2. **No Graph Metrics**
   - Verify CALLS/CALLED_BY relationships exist
   - Check method naming consistency
   - Validate graph connectivity

3. **Empty Results**
   - Check AST CSV format
   - Verify required columns exist
   - Review file path accuracy

4. **CRLF Line Ending Issues**
   - The script is created with LF line endings
   - Git should handle this automatically
   - If issues persist, check `.gitattributes` configuration

### Debug Mode

Enable verbose logging for debugging:

```python
analyzer = StaticImportanceAnalyzer(
    neo4j_uri="your_uri",
    neo4j_username="username",
    neo4j_password="password",
    verbose=True  # Enable detailed logging
)
```

## Performance

Typical performance on a standard machine:
- 1,000 methods: ~30 seconds
- 10,000 methods: ~5 minutes
- 100,000 methods: ~45 minutes

Performance scales primarily with:
- Graph connectivity (number of relationships)
- Neo4j query response time
- NetworkX centrality calculations

## Integration with Other Systems

This script is designed to be easily integrated with other Python applications:

```python
# Import and use directly
from static_importance_analyzer import StaticImportanceAnalyzer

# Use in your application
def analyze_codebase(ast_file_path, output_path):
    analyzer = StaticImportanceAnalyzer(
        neo4j_uri="your_uri",
        neo4j_username="username",
        neo4j_password="password"
    )
    
    results = analyzer.analyze(ast_file_path)
    if results is not None:
        analyzer.export_results(results, output_path)
        return True
    return False
```

## License

This project is available under the MIT License.

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
1. Check this README
2. Review test cases for examples
3. Enable verbose mode for debugging
4. Check Neo4j connectivity first
