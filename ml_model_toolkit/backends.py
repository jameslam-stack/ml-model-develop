"""DataFrame backend abstraction.

Supports both pandas and PySpark DataFrames through a common interface.
PySpark is imported lazily so the package works without it installed.
"""

from typing import Any, List
import numpy as np


class DataFrameBackend:
    """Abstract backend describing the operations the pipeline needs."""

    name = "backend"

    @property
    def is_pandas(self) -> bool:
        return False

    @property
    def is_pyspark(self) -> bool:
        return False

    def to_pandas(self, df: Any) -> Any:
        """Materialise a backend frame to pandas (used by sklearn-based steps)."""
        return df

    def columns(self, df: Any) -> List[str]:
        raise NotImplementedError

    def numeric_columns(self, df: Any) -> List[str]:
        raise NotImplementedError

    def select(self, df: Any, cols: List[str]) -> Any:
        raise NotImplementedError

    def drop(self, df: Any, cols: List[str]) -> Any:
        raise NotImplementedError

    def nunique(self, df: Any, col: str) -> int:
        raise NotImplementedError

    def isnull_sum(self, df: Any, col: str) -> int:
        raise NotImplementedError

    def fillna(self, df: Any, col: str, value: Any) -> Any:
        raise NotImplementedError

    def with_column(self, df: Any, name: str, value: Any) -> Any:
        raise NotImplementedError


class PandasBackend(DataFrameBackend):
    """Backend for pandas DataFrames (default)."""

    name = "pandas"

    @property
    def is_pandas(self) -> bool:
        return True

    def to_pandas(self, df):
        return df

    def columns(self, df):
        return list(df.columns)

    def numeric_columns(self, df):
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def select(self, df, cols):
        return df[cols]

    def drop(self, df, cols):
        return df.drop(columns=cols)

    def nunique(self, df, col):
        return int(df[col].nunique(dropna=True))

    def isnull_sum(self, df, col):
        return int(df[col].isnull().sum())

    def fillna(self, df, col, value):
        df[col] = df[col].fillna(value)
        return df

    def with_column(self, df, name, value):
        df[name] = value
        return df


class PySparkBackend(DataFrameBackend):
    """Backend for PySpark DataFrames (lazy import)."""

    name = "pyspark"

    @property
    def is_pyspark(self) -> bool:
        return True

    def __init__(self):
        try:
            from pyspark.sql import functions as F  # noqa: F401
            self._F = F
        except ImportError as exc:  # pragma: no cover - exercised only without pyspark
            raise ImportError(
                "PySpark backend requires pyspark. Install it with `pip install pyspark`."
            ) from exc

    def to_pandas(self, df):
        return df.toPandas()

    def columns(self, df):
        return df.columns

    def numeric_columns(self, df):
        from pyspark.sql.types import NumericType
        return [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

    def select(self, df, cols):
        return df.select(cols)

    def drop(self, df, cols):
        return df.drop(*cols)

    def nunique(self, df, col):
        return df.select(col).distinct().count()

    def isnull_sum(self, df, col):
        return df.filter(df[col].isNull()).count()

    def fillna(self, df, col, value):
        return df.na.fill(value, [col])

    def with_column(self, df, name, value):
        return df.withColumn(name, value)


def get_backend(df: Any) -> DataFrameBackend:
    """Detect the backend for a given DataFrame."""
    if df is None:
        return PandasBackend()
    module = type(df).__module__ or ""
    if module.startswith("pyspark") or "pyspark" in module:
        return PySparkBackend()
    return PandasBackend()


def is_pyspark(df: Any) -> bool:
    return get_backend(df).is_pyspark


__all__ = [
    "DataFrameBackend",
    "PandasBackend",
    "PySparkBackend",
    "get_backend",
    "is_pyspark",
]