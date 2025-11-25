# sql_queries.py
# Helper to load SQL scripts from the sqlScripts folder
import os

SQL_DIR = os.path.join(os.path.dirname(__file__), 'sqlScripts')

def load_sql(filename):
    path = os.path.join(SQL_DIR, filename)
    with open(path, encoding='utf-8') as f:
        return f.read()

# Example usage:
# query = load_sql('tender_table.sql')
