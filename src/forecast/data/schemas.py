"""Pydantic models for data validation and serialization."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class TimeSeriesRecord(BaseModel):
    """A single time series observation."""

    date: date
    product_id: str = Field(..., description="Product or revenue stream identifier")
    revenue: float = Field(..., ge=0, description="Daily revenue in USD")
    units_sold: int = Field(..., ge=0, description="Number of units sold")
    price: float = Field(..., gt=0, description="Unit price in USD")


class DatasetMetadata(BaseModel):
    """Metadata describing a generated or loaded dataset."""

    num_records: int
    num_products: int
    date_start: date
    date_end: date
    products: list[str]
    generated_at: Optional[str] = None


class ValidationReport(BaseModel):
    """Results from data validation checks."""

    is_valid: bool
    total_records: int
    missing_values: dict[str, int] = Field(default_factory=dict)
    duplicate_count: int = 0
    date_gaps: list[str] = Field(default_factory=list)
    outlier_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
