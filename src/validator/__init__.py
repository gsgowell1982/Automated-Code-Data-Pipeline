"""
Validator module for data quality and schema validation.

This module provides:
- Schema validation for generated training samples
- Data quality checks
- DBR alignment verification
"""

from .schema_validator import (
    SchemaValidator,
    TrainingSample,
    Context,
    AutoProcessing,
    DataQuality,
    ValidationResult,
)

__all__ = [
    "SchemaValidator",
    "TrainingSample",
    "Context",
    "AutoProcessing",
    "DataQuality",
    "ValidationResult",
]
