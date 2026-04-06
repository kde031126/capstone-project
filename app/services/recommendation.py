import random
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.models.record import Record
from app.models.phoneme_error import PhonemeError
from app.models.word import Word
from datetime import datetime, timedelta

def get_personalized_word_list(db: Session, user_id: int, current_step_id: int, limit: int = 10):
    """
    개인화 알고리즘: 신규(70%) : 복습(30%) 비율로 단어 추천
    오류 음소, 위치, 최근성, 반복성 가중치 계산
    """
    
    # 1. 아이의 최근 오답 패턴 분석 (취약 음소 및 위치 파악)
    # 최근 2주간 가장 많이 틀린 음소와 위치 조합을 가져옴
    two_weeks_ago = datetime.now() - timedelta(days=14)
    
    vulnerable_patterns = db.query(
        PhonemeError.target_phoneme,
        PhonemeError.error_position,
        func.count(PhonemeError.error_id).label('error_count')
    ).join(Record).filter(
        Record.user_id == user_id,
        Record.created_at >= two_weeks_ago
    ).group_by(
        PhonemeError.target_phoneme, 
        PhonemeError.error_position
    ).order_by(desc('error_count')).all()

    # 취약 패턴 리스트 (예: [('ㄱ', 2), ('ㅅ', 0)])
    patterns = [(p.target_phoneme, p.error_position) for p in vulnerable_patterns]

    # 2. 복습 단어 선정 (30% -> 3개)
    # 최근에 틀렸던(is_correct=False) 단어들 중 취약 패턴이 포함된 단어 우선
    review_count = int(limit * 0.3)
    review_words = []
    
    # 최근 오답 기록이 있는 단어 ID 추출
    past_wrong_word_ids = db.query(Record.word_id).filter(
        Record.user_id == user_id,
        Record.is_correct == False
    ).distinct().all()
    past_wrong_word_ids = [w[0] for w in past_wrong_word_ids]

    if past_wrong_word_ids:
        # 취약 패턴이 포함된 과거 오답 단어 필터링 (최근성/반복성 가중치)
        review_candidates = db.query(Word).filter(Word.word_id.in_(past_wrong_word_ids)).all()
        
        # 알고리즘: 취약 패턴과 일치하는 단어에 점수 부여 (간단한 정렬)
        review_candidates.sort(key=lambda w: any(p[0] in w.phonemes for p in patterns), reverse=True)
        review_words = review_candidates[:review_count]

    # 3. 신규 단어 선정 (70% -> 7개)
    # 한 번도 안 한 단어 중 현재 Step 또는 직후 Step 단어 우선
    new_count = limit - len(review_words)
    
    # 이미 학습한 단어 ID 제외
    learned_word_ids = db.query(Record.word_id).filter(Record.user_id == user_id).distinct().all()
    learned_word_ids = [w[0] for w in learned_word_ids]

    new_candidates = db.query(Word).filter(
        ~Word.word_id.in_(learned_word_ids),
        Word.step_id >= current_step_id  # 현재 또는 다음 스텝
    ).order_by(Word.step_id).limit(50).all() # 후보군 확보

    # 신규 단어 중에서도 취약 음소가 포함된 단어를 우선적으로 섞음
    random.shuffle(new_candidates)
    new_candidates.sort(key=lambda w: any(p[0] in w.phonemes for p in patterns), reverse=True)
    new_words = new_candidates[:new_count]

    # 4. 최종 리스트 합치기 및 셔플
    final_list = review_words + new_words
    random.shuffle(final_list)
    
    return [w.word_id for w in final_list]