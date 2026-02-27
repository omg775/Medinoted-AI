import sqlite3
import os

DB_PATH = os.path.join("data", "health_assistant.db")

def migrate():
    if not os.path.exists(DB_PATH):
        print("DB doesn't exist yet, nothing to migrate.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    try:
        # Check if role column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "role" not in columns:
            print("Adding 'role' column to users table...")
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'patient'")
            conn.commit()
            print("Migration successful.")
        else:
            print("'role' column already exists.")
    except Exception as e:
        print(f"Migration error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
