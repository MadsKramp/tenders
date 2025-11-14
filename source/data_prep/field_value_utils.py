""" Field value preparation module for spend analysis. Ensures Purchase Amount fields are displayed correctly.
Example: 1234567.89 -> 1,234,567.89 EUR """
def fmt_eur(value):
    """Format a numeric value as Euro currency with commas and EUR suffix.

    Args:
        value (float|int): Numeric value to format.

    Returns:
        str: Formatted string in Euro currency format.
    """
    try:
        formatted_value = f"{value:,.2f} EUR"
        return formatted_value
    except (ValueError, TypeError):
        return "Invalid value"
def parse_eur(formatted_str):
    """Parse a Euro formatted string back to a numeric value.

    Args:
        formatted_str (str): Formatted Euro string (e.g., "1,234,567.89 EUR").

    Returns:
        float: Numeric value.
    """
    try:
        # Remove ' EUR' suffix and commas
        numeric_str = formatted_str.replace(" EUR", "").replace(",", "")
        return float(numeric_str)
    except (ValueError, AttributeError):
        return None
def is_valid_eur_format(formatted_str):
    """Check if a string is in valid Euro currency format.

    Args:
        formatted_str (str): String to check.
    Returns:
        bool: True if valid Euro format, False otherwise.
    """
    import re
    pattern = r'^\d{1,3}(,\d{3})*(\.\d{2})? EUR$'
    return bool(re.match(pattern, formatted_str))
def add_eur_suffix(value):
    """Add EUR suffix to a numeric value.

    Args:
        value (float|int): Numeric value.

    Returns:
        str: String with EUR suffix.
    """
    return f"{value} EUR"

""" We also ensure that Purchase quantity fields are displayed correctly.
Example: 1234567.89 -> 1,234,567.89 units """
def fmt_units(value):
    """Format a numeric value as units with commas and 'units' suffix.

    Args:
        value (float|int): Numeric value to format.

    Returns:
        str: Formatted string in units format.
    """
    try:
        formatted_value = f"{value:,.2f} units"
        return formatted_value
    except (ValueError, TypeError):
        return "Invalid value"
def parse_units(formatted_str):
    """Parse a units formatted string back to a numeric value.

    Args:
        formatted_str (str): Formatted units string (e.g., "1,234,567.89 units").

    Returns:
        float: Numeric value.
    """
    try:
        # Remove ' units' suffix and commas
        numeric_str = formatted_str.replace(" units", "").replace(",", "")
        return float(numeric_str)
    except (ValueError, AttributeError):
        return None
def is_valid_units_format(formatted_str):
    """Check if a string is in valid units format.

    Args:
        formatted_str (str): String to check.
    Returns:
        bool: True if valid units format, False otherwise.
    """
    import re
    pattern = r'^\d{1,3}(,\d{3})*(\.\d{2})? units$'
    return bool(re.match(pattern, formatted_str))
