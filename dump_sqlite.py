import sqlite3
import pandas as pd

conn = sqlite3.connect('/tmp/trace_api.nsys-rep.sqlite')
# Get all CUDA API calls and Kernel executions sorted by start time
query = """
SELECT start, name_id.value as name
FROM CUPTI_ACTIVITY_KIND_RUNTIME runtime
LEFT JOIN StringIds name_id ON runtime.nameId = name_id.id
WHERE name LIKE '%cudaLaunchKernel%' OR name LIKE '%cudaMemsetAsync%' OR name LIKE '%cudaMemcpyAsync%'
ORDER BY start
"""
df = pd.read_sql_query(query, conn)
for i in range(len(df)):
    if 'cudaMemsetAsync' in df.iloc[i]['name']:
        print(f"--- MEMSET DETECTED ---")
        start = max(0, i-2)
        end = min(len(df), i+3)
        for j in range(start, end):
            print(f"[{j}] {df.iloc[j]['name']}")
        print("-----------------------")
