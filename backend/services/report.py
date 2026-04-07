import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.orm import Session
from models.record import Record
from models.phoneme_error import PhonemeError
from collections import Counter

# .env 파일 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_parent_report(db: Session, session_id: int):
    # 1. 해당 세션의 오답 데이터 수집
    records = db.query(Record).filter(Record.session_id == session_id).all()
    errors = db.query(PhonemeError).join(Record).filter(Record.session_id == session_id).all()
    
    if not records:
        return "데이터가 부족하여 리포트를 생성할 수 없습니다."

    # 2. 통계 계산 (정답률, 취약 음소)
    total_count = len(records)
    correct_count = sum(1 for r in records if r.is_correct)
    accuracy = (correct_count / total_count) * 100
    
    error_list = [e.target_phoneme for e in errors if e.target_phoneme]
    most_common = Counter(error_list).most_common(1)
    weak_point = most_common[0][0] if most_common else "없음"

    # 3. OpenAI GPT에게 리포트 요청
    prompt = f"""
    아동 언어 발달 서비스 '쑥쑥'의 인공지능 선생님으로서 부모님께 리포트를 작성해줘.
    
    [오늘의 학습 통계]
    - 학습 단어 수: {total_count}개
    - 정확도: {accuracy:.1f}%
    - 집중 교정이 필요한 음소: '{weak_point}'
    
    [요청 사항]
    1. 아이의 노력을 칭찬하며 따뜻한 어조로 시작할 것.
    2. '{weak_point}' 발음이 안 될 때 집에서 놀이처럼 연습할 수 있는 간단한 팁을 하나 제안할 것.
    3. 3~4문장 내외로 친절하게 작성할 것.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # 혹은 gpt-4o
        messages=[{"role": "system", "content": "너는 다정한 언어치료 전문가야."},
                  {"role": "user", "content": prompt}]
    )

    return {
        "summary": {
            "accuracy": accuracy,
            "weak_point": weak_point
        },
        "ai_message": response.choices[0].message.content
    }