# ML Model Developer Pipeline

A machine-learning model development pipeline implementing the
`agent-data-ml-model` workflow (v1.0.0):

**data analysis → preprocessing → model development → evaluation → deployment prep**

Supports pandas and PySpark DataFrames via a backend abstraction.

## Installation

Install directly from GitHub (clone the repo, then install in editable mode):

```bash
# 1. Clone the repo
git clone https://github.com/jameslam-stack/ml-model-develop.git
cd ml-model-develop

# 2. Install (editable mode — no PyPI required)
pip install -e .

# 3. Install dependencies
pip install scikit-learn pandas numpy
```

> **Note:** This package is not on PyPI. Use the GitHub install above.
> PySpark is optional — the pipeline falls back to pandas automatically.

### Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn` (installed automatically)
- `pyspark` (optional, for PySpark DataFrames)

## Quick Start

The pipeline auto-detects whether your target is **classification** or
**regression** — no configuration needed.

```python
import pandas as pd
from ml_model_toolkit import MLDeveloperPipeline

df = pd.read_csv('your_data.csv')

# Classification (binary/multi-class target)
pipeline = MLDeveloperPipeline(df, target_col='target')
pipeline.run_full()  # analysis → preprocess → model → evaluate → deploy

# Regression (continuous target)
pipeline = MLDeveloperPipeline(df, target_col='house_price')
pipeline.run_full()

print(pipeline.problem_type)  # 'classification' or 'regression'
print(pipeline.metrics['accuracy'])  # or 'r2_score' for regression
```

### Classification vs Regression

| Problem type | Detected from | Algorithms | Metrics |
|--------------|---------------|------------|---------|
| Classification | ≤20 unique target values | logistic_regression, random_forest, gradient_boosting, adaboost, svm_rbf, knn, naive_bayes | accuracy, confusion_matrix, roc_auc |
| Regression | continuous target | linear_regression, ridge, lasso, random_forest_reg, gradient_boosting_reg, svm_rbf_reg | r2_score, rmse, mae |

The problem type is auto-detected from the target column — no flag needed.

### Model Comparison & Cross-Validation

`build_model` supports 7 algorithms (classification) or 6 (regression) and
benchmarks them with cross-validation. `compare_models` picks the best.

```python
pipeline.compare_models()  # benchmark all algorithms
print(pipeline.metrics['best_algorithm'])
print(pipeline.metrics['model_comparison'])
```

Or run the full workflow with automatic comparison:

```python
pipeline.run_full(compare=True)  # trains all, picks best
```

Customise cross-validation folds:

```python
pipeline.build_model(algorithm='random_forest', cv_folds=10)
```

## Workflow Stages

```python
pipeline = MLDeveloperPipeline(df, target_col='target')

pipeline.analyze_data()     # 1. Data Analysis — EDA, stats, missing %
pipeline.preprocess()       # 2. Preprocessing — split, scale, impute
pipeline.build_model()      # 3. Model Development — logistic regression / random forest
pipeline.evaluate_model()   # 4. Evaluation — accuracy, confusion, ROC/AUC
pipeline.save_model()       # 5. Deployment Prep — pickle + JSON report
```

Or run the full workflow in one call:

```python
pipeline.run_full()
```

### PySpark DataFrames

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml').getOrCreate()
df = spark.read.csv('data.csv', header=True)

pipeline = MLDeveloperPipeline(df, target_col='target')
pipeline.run_full()  # backend detected automatically
```

sklearn-based steps convert the PySpark frame to pandas internally and
write results back to the PySpark format on save.

## Configuration

```python
pipeline = MLDeveloperPipeline(
    df,
    target_col='target',
    config={'model_path': 'models/my_model.pkl', 'report_path': 'reports/my_report.json'}
)
```

## Project Structure

```
ml-model-develop/
├── ml_model_toolkit/    # Source package
│   ├── __init__.py      # Package exports
│   ├── pipeline.py      # ML workflow (5 stages)
│   └── backends.py      # pandas / PySpark backend abstraction
├── tests/               # Test suite
├── examples/            # Tutorial notebooks
├── data/                # Sample data
├── models/              # Saved models
├── reports/             # Generated reports
├── README.md
└── pyproject.toml
```

## Examples

See `examples/ml_pipeline.ipynb` for a beginner-friendly walkthrough.

## License

MIT License — see LICENSE file.
