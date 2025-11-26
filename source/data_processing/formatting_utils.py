import re

def safe_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    name = str(name)
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def write_excel_with_thousands(
    df,
    path,
    sheet_name="Sheet1",
    index=False,
    thousand_cols=None,
):
    import pandas as pd

    if thousand_cols is None:
        thousand_cols = []

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        thousand_format = workbook.add_format({"num_format": "#,##0"})
        for col_name in thousand_cols:
            if col_name in df.columns:
                col_idx = df.columns.get_loc(col_name)
                worksheet.set_column(col_idx + 1, col_idx + 1, None, thousand_format)
