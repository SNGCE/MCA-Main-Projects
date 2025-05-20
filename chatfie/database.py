import sqlite3

# Function to connect to the database
def connect_db():
    return sqlite3.connect("chatfie.db")

def create_tables():
    with connect_db() as conn:
        cursor = conn.cursor()

        # ✅ User Registration Table (with profile picture support)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                gender TEXT NOT NULL,
                profile_pic TEXT DEFAULT 'static/default_profile.png'
            )
        ''')

        # ✅ Chat History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (user_id) REFERENCES registration (id)
            )
        ''')

        # ✅ Mood Tracking Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mood_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                mood TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES registration (id)
            )
        ''')

        # ✅ User Profile Update Table (Optional for logging updates)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profile_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                field_updated TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES registration (id)
            )
        ''')

        # ✅ sessions Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_title TEXT,
                FOREIGN KEY (user_id) REFERENCES registration(id)
            )
        ''')

        # ✅ feedback Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                rating INTEGER,
                feedback TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')


        conn.commit()

    print("✅ Database and tables created successfully!")

# Run table creation when script executes
if __name__ == "__main__":
    create_tables()
