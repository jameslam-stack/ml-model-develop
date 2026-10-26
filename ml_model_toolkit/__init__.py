"""ML Model Developer Pipeline — agent-data-ml-model (v1.0.0)."""
from .pipeline import MLDeveloperPipeline
from .backends import get_backend, is_pyspark
__version__ = "1.0.0"
__all__ = ["MLDeveloperPipeline", "get_backend", "is_pyspark"]
