"""Field schema for the supplied invoice image and scanned invoice PDF."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class InvoiceLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str = Field(min_length=1)
    bill_to: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    due_date: date
    line_items: list[InvoiceLine] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax_rate_percent: float | None = Field(default=None, ge=0)
    tax_amount: float = Field(ge=0)
    total_due: float = Field(ge=0)
    currency: str | None = None  # Not stated on these invoices; should remain null.
