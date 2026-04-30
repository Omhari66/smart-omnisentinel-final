import sqlite3
conn = sqlite3.connect('demo.db')
# Check incidents table columns
rows = conn.execute("PRAGMA table_info(incidents)").fetchall()
for r in rows:
    print(f"  {r[1]:25s} {r[2]}")

print("\n--- review_actions columns ---")
rows = conn.execute("PRAGMA table_info(review_actions)").fetchall()
for r in rows:
    print(f"  {r[1]:25s} {r[2]}")
conn.close()
