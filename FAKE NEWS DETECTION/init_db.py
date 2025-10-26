import sqlite3

conn = sqlite3.connect('logs.db')
conn.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input TEXT,
        result TEXT,
        timestamp DATETIME
    )
''')
conn.commit()
conn.close()
