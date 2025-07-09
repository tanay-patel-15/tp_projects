import sqlite3

# Step 1: Create an in-memory database and a table
conn = sqlite3.connect(':memory:')  # Temporary DB in RAM
cursor = conn.cursor()
cursor.execute("CREATE TABLE users (id INTEGER, name TEXT)")
cursor.execute("INSERT INTO users VALUES (1, 'Alice')")
cursor.execute("INSERT INTO users VALUES (2, 'Bob')")
conn.commit()

# Step 2: WITHOUT row_factory (default behavior)
print("Without row_factory:")
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(row[0], row[1])  # Access using index

# Step 3: WITH row_factory (dictionary-style)
print("\nWith row_factory:")
conn.row_factory = sqlite3.Row  # Make rows act like dictionaries
cursor = conn.cursor()
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(row['id'], row['name'])  # Access using column names

conn.close()
