"""
Domain-level exceptions.
Raised inside use-cases; caught at the API boundary.
"""


class DomainError(Exception):
    """Base class for all domain errors."""


class ValidationError(DomainError):
    """Input data failed schema or business-rule validation."""


class ModelNotLoadedError(DomainError):
    """Attempted inference before model artifacts were loaded."""


class InferenceError(DomainError):
    """Runtime error during model inference."""


class RepositoryError(DomainError):
    """Error loading or saving model artifacts."""
