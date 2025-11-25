import re

def safe_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    name = str(name)
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def write_excel_with_thousands(writer, df, sheet_name):
    """Stub: Write DataFrame to Excel with thousands separator formatting (customize as needed)."""
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    # You can add formatting logic here if needed.
