from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as db_session
from sqlalchemy.orm import joinedload 
from datetime import datetime
from typing import List, Optional

from db.session import get_db
from models.record import Record
from models.practice_session import PracticeSession 
from schemas.practice_session import SessionCreate, SessionResponse
from services.recommendation import get_personalized_word_list
from services import report  # [확인] 리포트 서비스

router = APIRouter(prefix="/sessions", tags=["Sessions"])

# 1. 세션 시작 (POST) - 그대로 유지
@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(session_in: SessionCreate, db: db_session = Depends(get_db)):
    word_ids = get_personalized_word_list(
        db, 
        user_id=session_in.user_id, 
        current_step_id=session_in.step_id,
        limit=session_in.words_count
    )
    
    if not word_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="해당 단계에서 추천할 수 있는 단어가 없습니다."
        )

    new_session = PracticeSession(
        user_id=session_in.user_id,
        step_id=session_in.step_id,
        target_words=word_ids,
        words_count=len(word_ids),
        is_completed=0,
        started_at=datetime.now()
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session

# 2. 세션 종료 (PATCH) - 그대로 유지
@router.patch("/{session_id}", response_model=SessionResponse)
def end_session(session_id: int, db: db_session = Depends(get_db)):
    session = db.query(PracticeSession).filter(PracticeSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    session.ended_at = datetime.now()
    session.is_completed = 1
    db.commit()
    db.refresh(session)
    return session

# 3. 세션 상세 조회 (GET) - [수정됨: 이름 중복 제거 및 리포트 합치기]
@router.get("/{session_id}")
def get_session_detail(session_id: int, db: db_session = Depends(get_db)):
    """
    특정 세션의 결과와 AI 리포트를 함께 조회합니다.
    """
    # 굴비 엮듯 한 번에 가져오기 (Eager Loading)
    session = db.query(PracticeSession)\
        .options(joinedload(PracticeSession.records).joinedload(Record.errors))\
        .filter(PracticeSession.session_id == session_id)\
        .first()

    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # [핵심] OpenAI를 이용한 리포트 데이터 생성
    # report.py에 만든 함수를 여기서 호출합니다!
    report_data = report.generate_parent_report(db, session_id)

    # 10개 중 몇 개 맞았는지 계산
    correct_count = sum(1 for r in session.records if r.is_correct)
    
    return {
        "session_info": {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "step_id": session.step_id,
            "is_completed": session.is_completed,
            "started_at": session.started_at,
            "ended_at": session.ended_at
        },
        "report": report_data,  # AI 메시지가 여기에 담겨요!
        "stats": {
            "total_words": len(session.records),
            "correct_words": correct_count,
            "accuracy_rate": (correct_count / len(session.records) * 100) if session.records else 0
        },
        "records": [
            {
                "record_id": r.record_id,
                "word_id": r.word_id,
                "is_correct": r.is_correct,
                "child_text": r.child_text,
                "child_phonemes": r.child_phonemes,
                "errors": r.errors 
            } for r in session.records
        ]
    }