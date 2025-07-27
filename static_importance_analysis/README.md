# Static Importance Analysis for Java Methods

This module computes static importance indices for Java methods using both Knowledge Graph data from Neo4j and AST metadata. The calculated weights are designed to be used in a hybrid RAG system for code generation and retrieval.

## Overview

The static importance analysis combines multiple metrics to rank Java methods by their significance in the codebase:

### Metrics Computed

#### Code Complexity Metrics (40% weight)
- **Lines of Code (LOC)**: 8%
- **Cyclomatic Complexity**: 12% 
- **Cognitive Complexity**: 10%
- **Halstead Effort**: 10%

#### Graph Centrality Metrics (35% weight)
- **Degree Centrality**: 10%
- **Betweenness Centrality**: 10%
- **Eigenvector Centrality**: 8%
- **Fan-in**: 4%
- **Fan-out**: 3%

#### Parameter and Interface Metrics (25% weight)
- **Parameter Count**: 8%
- **Parameter Complexity**: 9%
- **Return Type Complexity**: 8%

## Apporach


### 🎯 Overall Approach
Our static importance calculation uses a multi-dimensional weighted scoring system that combines different aspects of code analysis to create a single, normalized importance score for each Java method.
### ⚖️ Weight Distribution Strategy
The weights are carefully designed to balance different aspects of method importance:

1. Code Complexity Metrics (35% total weight): 
- LOC (7%): Lines of code indicate method size/complexity

- Cyclomatic Complexity (10%): Control flow complexity
- Cognitive Complexity (8%): Human comprehension difficulty
- Halstead Effort (10%): Computational complexity
2. Graph Centrality Metrics (30% total weight):
- Degree Centrality (8%): Overall connectivity in call graph
- Betweenness Centrality (8%): Methods that bridge different parts
- Eigenvector Centrality (6%): Connected to important methods
- Fan-in (4%): How many methods call this one
- Fan-out (4%): How many methods this one calls
3. Parameter & Interface Metrics (20% total weight):
- Parameter Count (7%): Interface complexity
- Parameter Complexity (7%): Complex data types used
- Return Type Complexity (6%): Return type sophistication
4. Relative Importance Metrics (15% total weight):
- Class Relative Importance (8%): Importance within the same class
- Name Similarity Importance (7%): Importance compared to similarly named methods

### 🔄 Normalization Process
#### Step 1: Individual Metric Normalization

1. Each raw metric is normalized to 0-1 range using appropriate methods:
- Min-Max Normalization: (value - min) / (max - min)
- Robust Normalization: For outlier-resistant scaling
- Capping: Extreme outliers are capped at 95th percentile

#### Step 2: Weighted Combination

`Final Score = Σ(normalized_metric_i × weight_i) for all metrics`

#### Step 3: Final Normalization

- The combined scores undergo robust normalization again
- This ensures meaningful distribution across the 0-1 range
- Handles any remaining outliers from the combination process

### 📊 Categorization System
Methods are categorized using percentile-based thresholds:

- Critical: Top 10% (≥90th percentile)
- High: 75th-90th percentile
- Medium: 50th-75th percentile
- Low: 25th-50th percentile
- Minimal: Bottom 25% (<25th percentile)

### 🎯 Design Rationale

1. Balanced Approach: No single metric dominates (largest weight is 10%)
Multi-Dimensional: Combines static complexity, graph structure, and context
Robust to Outliers: Uses capping and robust normalization techniques
Relative Scoring: Considers importance within class and similarity groups
Interpretable: Clear percentile-based categories for practical use
💡 Key Benefits

Prevents Bias: No metric can dominate due to large raw values
Maintains Meaning: Relative differences within metrics are preserved
Enables Combination: Different metric types can be meaningfully combined
Supports Interpretation: Final scores have clear meaning across methods
🔧 Practical Application
These normalized importance scores can be used in your hybrid RAG system to:

Weight method retrieval results based on static importance
Prioritize which methods to include in context windows
Focus code generation on high-importance architectural patterns
Filter out noise from low-importance utility methods
The normalization ensures that a method with high complexity (e.g., LOC=100) and high centrality (e.g., degree=0.8) gets appropriately weighted contributions from both metrics, rather than being dominated by the raw LOC value. This creates a fair, interpretable scoring system that captures the true importance of methods in your codebase.


## Data Sources

- **Neo4j Knowledge Graph**: `http://4.187.169.27:7474/browser/`
  - Username: `neo4j`
  - Password: `MyStrongPassword123`
- **AST Data**: `../AST/java_parsed.csv`
- **Target Project**: Library Management System

## Files

- `static_importance_calculator.ipynb`: Main Jupyter notebook for analysis
- `requirements.txt`: Python dependencies
- `enhanced_java_methods_with_importance.csv`: Output dataset (generated)
- `analysis_summary.json`: Analysis statistics (generated)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Neo4j database is accessible and contains the knowledge graph data

3. Verify the AST CSV file exists at `../AST/java_parsed.csv`

4. Run the Jupyter notebook:
```bash
jupyter notebook static_importance_calculator.ipynb
```

## Output

The analysis produces:

1. **Enhanced Dataset** (`enhanced_java_methods_with_importance.csv`):
   - All original AST data
   - Computed complexity metrics
   - Graph centrality measures
   - Normalized importance scores (0-1)
   - Importance categories (Critical, High, Medium, Low, Minimal)

2. **Analysis Summary** (`analysis_summary.json`):
   - Dataset overview statistics
   - Average metric values
   - Importance distribution

## Importance Categories

Methods are categorized based on their normalized importance score:
- **Critical**: ≥ 0.8 (Top-tier methods)
- **High**: 0.6 - 0.8 (Important methods)
- **Medium**: 0.4 - 0.6 (Moderately important)
- **Low**: 0.2 - 0.4 (Less important)
- **Minimal**: < 0.2 (Minimal importance)

## Usage in Hybrid RAG

The `importance_score_normalized` column can be used as weights in your retrieval system:

```python
# Example usage in retrieval
method_weights = df.set_index('Method Name')['importance_score_normalized'].to_dict()

# Apply weights during similarity search
weighted_similarity = base_similarity * method_weights.get(method_name, 0.5)
```

## Customization

To adjust the importance weights, modify the `weights` dictionary in the `StaticImportanceCalculator` class:

```python
self.weights = {
    'LOC': 0.08,                    # Adjust these values
    'Cyclomatic_Complexity': 0.12,  # to change metric importance
    # ... other weights
}
```

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key libraries:
- `pandas`, `numpy`: Data manipulation
- `neo4j`: Knowledge graph connectivity  
- `networkx`: Graph analysis
- `matplotlib`, `seaborn`: Visualization
- `tree-sitter`: Enhanced AST parsing (optional)

## Troubleshooting

### Neo4j Connection Issues
- Verify the Neo4j server is running
- Check network connectivity to `4.187.169.27:7687`
- Confirm credentials are correct

### Missing AST Data
- Ensure the AST parser has been run on the Library Assistant project
- Check that `../AST/java_parsed.csv` exists and is readable

### Performance Issues
- For large codebases, consider sampling methods for centrality calculation
- Use approximate algorithms for betweenness centrality on large graphs
