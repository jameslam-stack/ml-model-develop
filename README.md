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

```python
import pandas as pd
from ml_model_toolkit import MLDeveloperPipeline

df = pd.read_csv('your_data.csv')

pipeline = MLDeveloperPipeline(df, target_col='target')
pipeline.run_full()  # analysis → preprocess → model → evaluate → deploy

print(pipeline.metrics['accuracy'])
pipeline.save_model()  # model + scaler + metrics -> models/ml_model.pkl
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
