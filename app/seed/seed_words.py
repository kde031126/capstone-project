from app.db.session import SessionLocal
from app.models.word import Word

db = SessionLocal()

words = [
    Word(word_text="사과", standard_phonemes="사과"),
    Word(word_text="바나나", standard_phonemes="바나나"),
]

db.add_all(words)
db.commit()
db.close()

print("Seed data inserted!")
