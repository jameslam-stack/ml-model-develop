"""ML Model pipeline — agent-data-ml-model workflow (v1.0.0).

Stages from SKILL.md:
1 Data Analysis  2 Preprocessing  3 Model Development
4 Evaluation     5 Deployment Prep
Supports pandas / PySpark via backend abstraction.
"""
import pickle, json, os, datetime
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, classification_report)

from .backends import get_backend

class MLDeveloperPipeline:
    """End-to-end ML model development matching agent-data-ml-model spec."""

    def __init__(self, df=None, target_col="target", config=None):
        self.df = df.copy() if df is not None else None
        self.target_col = target_col
        self.backend = get_backend(df)
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.predictions = None
        self.metrics = {}
        self.log = []
        self.model_path = self.config.get("model_path", "models/ml_model.pkl")
        self.report_path = self.config.get("report_path", "reports/ml_report.json")

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log.append(entry); print(msg)
        return self

    # 1. DATA ANALYSIS
    def analyze_data(self):
        """EDA: stats, shape, missing %, feature types."""
        if self.backend.is_pyspark:
            self.df = self.backend.to_pandas(self.df)
            self.backend = get_backend(self.df)
            self._log("analyze_data: PySpark -> pandas conversion")
        self._log(f"Data Analysis — shape={self.df.shape}, target={self.target_col}")
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric: numeric.remove(self.target_col)
        missing = self.df.isnull().sum().to_dict()
        self.metrics["missing_summary"] = {k: int(v) for k, v in missing.items() if v > 0}
        self.metrics["numeric_features"] = numeric[:10]
        self._log(f"Numeric features ({len(numeric)}): {numeric[:5]}...")
        return self

    # 2. PREPROCESSING
    def preprocess(self):
        """Split train/test; scale; handle missing; encode target."""
        if self.backend.is_pyspark:
            self.df = self.backend.to_pandas(self.df)
            self.backend = get_backend(self.df)
            self._log("preprocess: PySpark -> pandas")
        df = self.df.drop_duplicates()
        # Numerical imputation (mean)
        numeric = df.select_dtypes(include=[np.number]).columns
        for col in numeric:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        # Feature / target split
        feature_cols = [c for c in df.columns if c != self.target_col]
        X = df[feature_cols].fillna(0)
        y = df[self.target_col]
        if y.dtype == object: y = pd.Categorical(y).codes
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        # Scaling
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train),
                                    columns=self.X_train.columns, index=self.X_train.index)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test),
                                   columns=self.X_test.columns, index=self.X_test.index)
        self._log("Preprocess: train/test split + StandardScaler applied")
        return self

    # 3. MODEL DEVELOPMENT
    def build_model(self, algorithm="logistic_regression", cv_folds=5):
        """Train model with algorithm selection and cross-validation.
        Algorithms: logistic_regression, random_forest, gradient_boosting,
        adaboost, svm_rbf, knn, naive_bayes.
        """
        algorithms = {
            "logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "adaboost": AdaBoostClassifier(random_state=42),
            "svm_rbf": SVC(kernel="rbf", random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "naive_bayes": GaussianNB(),
        }
        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(algorithms)}")
        self.model = algorithms[algorithm]
        # Cross-validation on training data
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train,
                                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                    scoring="accuracy")
        self.metrics["cv_accuracy_mean"] = round(float(cv_scores.mean()), 4)
        self.metrics["cv_accuracy_std"] = round(float(cv_scores.std()), 4)
        self.model.fit(self.X_train, self.y_train)
        self._log(f"Model Development — {algorithm} trained; CV accuracy: {self.metrics['cv_accuracy_mean']:.3f} ± {self.metrics['cv_accuracy_std']:.3f}")
        return self

    # 4. EVALUATION
    def evaluate_model(self):
        """Compute accuracy, confusion matrix, ROC/AUC, report."""
        y_pred = self.model.predict(self.X_test)
        self.predictions = y_pred
        acc = accuracy_score(self.y_test, y_pred)
        self.metrics["accuracy"] = round(float(acc), 4)
        self.metrics["confusion_matrix"] = confusion_matrix(self.y_test, y_pred).tolist()
        try:
            if hasattr(self.model, "predict_proba"):
                scores = self.model.predict_proba(self.X_test)[:, 1]
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(self.X_test)
            else:
                scores = y_pred
            self.metrics["roc_auc"] = round(float(roc_auc_score(self.y_test, scores)), 4)
        except Exception:
            self.metrics["roc_auc"] = None
        self._log(f"Evaluation — Accuracy: {acc:.3f}")
        # Classification report (string)
        self.metrics["classification_report"] = classification_report(
            self.y_test, y_pred, digits=3, output_dict=True)
        return self

    # 5. DEPLOYMENT PREP
    def save_model(self, path=None):
        """Serialize model + scaler + metrics to disk."""
        path = path or self.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "log": self.log,
            "target_col": self.target_col,
            "created_at": datetime.datetime.now().isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        self._log(f"Deployment Prep — model saved to {path}")
        return self

    def generate_report(self):
        return {
            "workflow": "agent-data-ml-model (v1.0.0)",
            "stages": ["analysis","preprocessing","development","evaluation","deployment"],
            "metrics": self.metrics,
            "log": self.log,
        }

    def save_report(self, path=None):
        path = path or self.report_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.generate_report(), f, indent=2, default=str)
        self._log(f"Report saved to {path}")
        return self

    def compare_models(self, algorithms=None, cv_folds=5):
        """Benchmark multiple algorithms with cross-validation and pick the best.

        Returns the winning algorithm name and stores all results in
        self.metrics['model_comparison'].
        """
        if algorithms is None:
            algorithms = ["logistic_regression", "random_forest",
                          "gradient_boosting", "adaboost",
                          "svm_rbf", "knn", "naive_bayes"]
        results = {}
        for algo in algorithms:
            try:
                self.build_model(algorithm=algo, cv_folds=cv_folds)
                results[algo] = {
                    "cv_accuracy_mean": self.metrics.get("cv_accuracy_mean"),
                    "cv_accuracy_std": self.metrics.get("cv_accuracy_std"),
                    "test_accuracy": self.metrics.get("accuracy"),
                }
            except Exception as e:
                self._log(f"compare_models: {algo} failed — {e}")
                results[algo] = {"error": str(e)}
        self.metrics["model_comparison"] = results
        # Pick best by CV mean
        best = max(results.items(),
                   key=lambda kv: kv[1].get("cv_accuracy_mean", -1))
        self.best_algorithm = best[0]
        self.metrics["best_algorithm"] = best[0]
        self._log(f"compare_models: best = {best[0]} (CV {best[1]['cv_accuracy_mean']:.3f})")
        return self

    def run_full(self, df=None, target_col="target", algorithm="logistic_regression",
                 compare=False, cv_folds=5):
        """Full ML workflow: 1→2→3→4→5.

        Parameters
        ----------
        df : DataFrame, optional
            Input data. If None, uses the frame passed at construction.
        target_col : str
            Name of the target column.
        algorithm : str
            Model algorithm to train.
        compare : bool
            If True, benchmark all algorithms and pick the best.
        cv_folds : int
            Number of cross-validation folds.
        """
        if df is not None:
            self.df = df.copy() if hasattr(df,"copy") else df
            self.backend = get_backend(self.df)
            self.target_col = target_col
        self.analyze_data()
        self.preprocess()
        if compare:
            self.compare_models(cv_folds=cv_folds)
            algorithm = getattr(self, "best_algorithm", algorithm)
            self.build_model(algorithm=algorithm, cv_folds=cv_folds)
        else:
            self.build_model(algorithm=algorithm, cv_folds=cv_folds)
        self.evaluate_model()
        self.save_model()
        self.save_report()
        self._log("=== ML MODEL WORKFLOW COMPLETE ===")
        return self

