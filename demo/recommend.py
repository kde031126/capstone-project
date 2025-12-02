import os
import sqlite3
import json
from collections import defaultdict
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "word_db.sqlite")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1. LearningLog 기반 오류 패턴 점수 계산
def get_error_pattern_scores(learning_log):
    pattern_scores = defaultdict(float)

    for log in learning_log:
        # 예: {"phoneme":"ㅅ","position":"초성","type":"교체","count":3,"date":"2025-11-30"}

        days_ago = (datetime.now() - datetime.fromisoformat(log["date"])).days
        recency_weight = max(0.5, 1 - days_ago / 30)

        score = log["count"] * recency_weight
        pattern_key = f"{log['position']}_{log['phoneme']}_{log['type']}"

        pattern_scores[pattern_key] += score

    return pattern_scores


# 2. Word DB에서 후보 단어 조회
def get_candidate_words(pattern_scores):
    candidates = {}

    cur.execute("SELECT word, initial_phonemes, medial_phonemes, final_phonemes FROM words")
    rows = cur.fetchall()

    for row in rows:
        word, initial, medial, final = row
        initial = json.loads(initial)
        medial = json.loads(medial)
        final = json.loads(final)

        for pattern in pattern_scores:
            pos, phoneme, _ = pattern.split("_")

            if (pos == "초성" and phoneme in initial) or \
               (pos == "중성" and phoneme in medial) or \
               (pos == "종성" and phoneme in final):

                if word not in candidates:
                    candidates[word] = {"patterns": [], "score": 0}

                candidates[word]["patterns"].append(pattern)

    return candidates


# 3. 후보 단어 점수 계산
def calculate_word_scores(candidates, pattern_scores):
    result = []

    for word, info in candidates.items():
        score = sum(pattern_scores[p] for p in info["patterns"])
        info["score"] = score
        result.append((word, info))

    # 점수 순 정렬
    result.sort(key=lambda x: x[1]["score"], reverse=True)
    return result


# 최종 추천 함수
def recommend_words(learning_log, top_n=10):
    pattern_scores = get_error_pattern_scores(learning_log)
    candidates = get_candidate_words(pattern_scores)
    ranked = calculate_word_scores(candidates, pattern_scores)
    return ranked[:top_n]


# 테스트용
if __name__ == "__main__":
    sample = [
        {"phoneme": "ㅅ", "position": "초성", "type": "교체", "count": 3, "date": "2025-11-30"},
        {"phoneme": "ㄱ", "position": "초성", "type": "탈락", "count": 2, "date": "2025-11-28"},
    ]

    results = recommend_words(sample)

    for w, info in results:
        print(w, info)
