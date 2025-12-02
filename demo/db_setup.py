import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "word_db.sqlite")

# DB 초기화
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 테이블 생성
cur.execute("""
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    initial_phonemes TEXT NOT NULL,
    medial_phonemes TEXT NOT NULL,
    final_phonemes TEXT NOT NULL
)
""")

# 사전 데이터 (원래 네가 만들었던 구성)
words = [
    {
        "word": "사과",
        "initial": ["ㅅ", "ㄱ"],
        "medial": ["ㅏ", "ㅘ"],
        "final": ["", ""]
    },
    {
        "word": "바나나",
        "initial": ["ㅂ", "ㄴ", "ㄴ"],
        "medial": ["ㅏ", "ㅏ", "ㅏ"],
        "final": ["", "", ""]
    },
    {
        "word": "학교",
        "initial": ["ㅎ", "ㄱ"],
        "medial": ["ㅏ", "ㅛ"],
        "final": ["ㄱ", ""]
    }
]

# 기존 데이터 삭제
cur.execute("DELETE FROM words")

# INSERT
for w in words:
    cur.execute("""
        INSERT INTO words (word, initial_phonemes, medial_phonemes, final_phonemes)
        VALUES (?, ?, ?, ?)
    """, (
        w["word"],
        json.dumps(w["initial"], ensure_ascii=False),
        json.dumps(w["medial"], ensure_ascii=False),
        json.dumps(w["final"], ensure_ascii=False),
    ))

conn.commit()
conn.close()
print("word_db.sqlite 초기화 완료!")
