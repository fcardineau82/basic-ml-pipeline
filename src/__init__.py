"""Package exports for src.

Avoid eager imports to prevent side effects when running submodules
directly (e.g., python -m src.ingestion).
"""

__all__ = ["run_ingestion_pipeline"]


def __getattr__(name):
    if name == "run_ingestion_pipeline":
        from .ingestion import run_ingestion_pipeline
        return run_ingestion_pipeline
    raise AttributeError(f"module 'src' has no attribute {name}")