import sqlite3
import json

DB_PATH = "gestures.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Gestures table
    c.execute('''CREATE TABLE IF NOT EXISTS gestures
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE,
                  action TEXT)''')
    # Samples table (landmarks stored as JSON)
    c.execute('''CREATE TABLE IF NOT EXISTS samples
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  gesture_id INTEGER,
                  landmarks TEXT,
                  FOREIGN KEY(gesture_id) REFERENCES gestures(id))''')
    conn.commit()
    conn.close()

def get_gestures():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, action FROM gestures")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "action": r[2]} for r in rows]

def add_gesture(name, action):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO gestures (name, action) VALUES (?, ?)", (name, action))
        conn.commit()
        gesture_id = c.lastrowid
    except sqlite3.IntegrityError:
        gesture_id = None
    conn.close()
    return gesture_id

def delete_gesture(gesture_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM samples WHERE gesture_id=?", (gesture_id,))
    c.execute("DELETE FROM gestures WHERE id=?", (gesture_id,))
    conn.commit()
    conn.close()

def add_sample(gesture_id, landmarks):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO samples (gesture_id, landmarks) VALUES (?, ?)",
              (gesture_id, json.dumps(landmarks)))
    conn.commit()
    conn.close()

def get_all_samples():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT gestures.name, samples.landmarks FROM samples JOIN gestures ON samples.gesture_id = gestures.id")
    rows = c.fetchall()
    conn.close()
    return [(name, json.loads(landmarks)) for name, landmarks in rows]

def update_gesture_action(gesture_id, action):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE gestures SET action=? WHERE id=?", (action, gesture_id))
    conn.commit()
    conn.close()