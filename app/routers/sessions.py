from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from app.db.session import get_db
from app.models.practice_session import PracticeSession
from app.schemas.practice_session import SessionCreate, SessionResponse

router = APIRouter()

# 1. 세션 시작 (POST)
@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(session_in: SessionCreate, db: Session = Depends(get_db)):
    """
    아이가 학습을 시작할 때 호출합니다. 
    새로운 세션 ID를 생성하고 시작 시간을 기록합니다.
    """
    # 팁: 혹시 유저나 단계가 실제로 존재하는지 체크하는 로직을 넣으면 더 안전해요!
    
    new_session = PracticeSession(
        user_id=session_in.user_id,
        step_id=session_in.step_id,
        words_count=session_in.words_count,
        started_at=datetime.now() # 서버 기준 현재 시간 저장
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session

# 2. 세션 종료 (PATCH)
@router.patch("/{session_id}", response_model=SessionResponse)
def end_session(session_id: int, db: Session = Depends(get_db)):
    """
    학습이 끝났을 때 호출하여 종료 시간을 기록합니다.
    """
    session = db.query(PracticeSession).filter(PracticeSession.session_id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    # 종료 시간 업데이트
    session.ended_at = datetime.now()
    
    db.commit()
    db.refresh(session)
    return session