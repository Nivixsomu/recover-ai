"""Read-only inspection of the official UCI dataset archives for RecoverAI."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "external"


def read_excel_from_archive(archive_name: str) -> tuple[str, pd.DataFrame]:
    """Load the single original Excel workbook from a UCI ZIP without extracting it."""
    archive_path = EXTERNAL_DATA_DIR / archive_name
    with zipfile.ZipFile(archive_path) as archive:
        workbook_name = next(
            name for name in archive.namelist() if name.lower().endswith((".xls", ".xlsx"))
        )
        workbook = BytesIO(archive.read(workbook_name))

    engine = "openpyxl" if workbook_name.lower().endswith(".xlsx") else "xlrd"
    header = 1 if workbook_name.lower().endswith(".xls") else 0
    return workbook_name, pd.read_excel(workbook, engine=engine, header=header)


def print_frame_overview(name: str, frame: pd.DataFrame) -> None:
    """Print the shared structural and numeric diagnostics for one dataset."""
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {len(frame.columns)}")
    print(f"Duplicate rows: {frame.duplicated().sum():,}")
    print("\nColumn names and data types:")
    for column, dtype in frame.dtypes.items():
        print(f"- {column}: {dtype}")

    missing = frame.isna().sum()
    print("\nMissing values:")
    for column, count in missing.items():
        print(f"- {column}: {count:,} ({count / len(frame) * 100:.2f}%)")

    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        print("\nNumeric summary statistics:")
        print(numeric.describe().round(2).to_string())


def print_value_counts(frame: pd.DataFrame, columns: list[str]) -> None:
    """Print the ten most common non-null values for selected categorical columns."""
    print("\nImportant categorical value counts (top 10):")
    for column in columns:
        if column in frame.columns:
            print(f"\n{column}:")
            print(frame[column].value_counts(dropna=False).head(10).to_string())


def inspect_online_retail() -> None:
    filename, frame = read_excel_from_archive("online+retail.zip")
    print(f"Original workbook inside archive: {filename}")
    print_frame_overview("UCI Online Retail", frame)
    print(f"\nUnique non-null CustomerID values: {frame['CustomerID'].nunique(dropna=True):,}")
    invoice_dates = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    print(f"InvoiceDate range: {invoice_dates.min()} to {invoice_dates.max()}")
    line_amount = pd.to_numeric(frame["Quantity"], errors="coerce") * pd.to_numeric(
        frame["UnitPrice"], errors="coerce"
    )
    print("\nDerived line amount (Quantity × UnitPrice) distribution:")
    print(line_amount.describe().round(2).to_string())
    print_value_counts(frame, ["Country", "InvoiceNo", "StockCode", "Description"])


def inspect_credit_default() -> None:
    filename, frame = read_excel_from_archive("default+of+credit+card+clients.zip")
    print(f"Original workbook inside archive: {filename}")
    print_frame_overview("UCI Default of Credit Card Clients", frame)
    print(f"\nUnique ID values: {frame['ID'].nunique(dropna=True):,}")
    print_value_counts(
        frame,
        [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "default payment next month",
        ],
    )


def main() -> None:
    """Inspect both datasets independently without writing any data files."""
    inspect_online_retail()
    inspect_credit_default()


if __name__ == "__main__":
    main()
