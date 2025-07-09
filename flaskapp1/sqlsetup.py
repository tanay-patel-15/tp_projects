# Run once to create the database
import sqlite3
conn = sqlite3.connect('database.db')
conn.execute('CREATE TABLE todos (id INTEGER PRIMARY KEY, task TEXT)')
conn.close()
