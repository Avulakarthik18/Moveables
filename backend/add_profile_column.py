import sqlite3
from datetime import datetime
import random

conn = sqlite3.connect("login.db")
cur = conn.cursor()

# -----------------------------
# 1️⃣ Create new table safely
# -----------------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS signup_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT UNIQUE,
    profile_img TEXT,
    created_at TEXT
)
""")

print("✅ New table ready")

# -----------------------------
# 2️⃣ Copy old data
# -----------------------------
cur.execute("""
INSERT INTO signup_new
(first_name,last_name,email,profile_img)
SELECT first_name,last_name,email,profile_img
FROM signup
""")

print("📦 Data copied")

# -----------------------------
# 3️⃣ Drop old table
# -----------------------------
cur.execute("DROP TABLE signup")
print("🗑 Old table removed")

# -----------------------------
# 4️⃣ Rename new table
# -----------------------------
cur.execute("""
ALTER TABLE signup_new
RENAME TO signup
""")

print("📦 Table renamed")

# -----------------------------
# 5️⃣ Add new columns safely
# -----------------------------
try:
    cur.execute("ALTER TABLE signup ADD COLUMN phone TEXT")
except:
    print("⚠️ phone column already exists")

try:
    cur.execute("ALTER TABLE signup ADD COLUMN account_id TEXT")
except:
    print("⚠️ account_id column already exists")

# -----------------------------
# 6️⃣ Add created date
# -----------------------------
today = datetime.now().strftime("%d %b %Y")

cur.execute("""
UPDATE signup
SET created_at = ?
WHERE created_at IS NULL
""", (today,))

print("📅 Dates added")

# -----------------------------
# 7️⃣ Generate Account IDs
# -----------------------------
users = cur.execute(
    "SELECT id FROM signup"
).fetchall()

for u in users:
    acc_id = "CRP" + str(random.randint(10000,99999))

    cur.execute("""
    UPDATE signup
    SET account_id = ?
    WHERE id = ?
    """, (acc_id, u[0]))

print("🆔 Account IDs generated")

cur.execute("DROP TABLE IF EXISTS signup_new")

# -----------------------------
conn.commit()
conn.close()

print("🚀 Migration completed successfully")