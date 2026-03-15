import sqlite3
for fname in ['trace_sum', 'trace_flat']:
    print(f"\n======== {fname} ========")
    try:
        conn = sqlite3.connect(f'/tmp/{fname}.sqlite')
        cursor = conn.cursor()
        
        cursor.execute("SELECT correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET")
        mem_ids = [r[0] for r in cursor.fetchall()]
        
        query = """
        SELECT runtime.correlationId, name_id.value as name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME runtime
        LEFT JOIN StringIds name_id ON runtime.nameId = name_id.id
        ORDER BY start
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for mem_id in mem_ids:
            for i, (cid, name) in enumerate(rows):
                if cid == mem_id:
                    print(f"--- MEMSET {cid} ---")
                    for j in range(max(0, i-2), min(len(rows), i+2)):
                        marker = ">> " if j == i else "   "
                        print(f"{marker}[{j}] {rows[j][1]}")
    except Exception as e:
        print(f"Error: {e}")
