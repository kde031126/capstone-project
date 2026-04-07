from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.session import get_db
from app.models.record import Record
from app.models.phoneme_error import PhonemeError
from app.models.word import Word
from app.models.practice_session import PracticeSession # user_id를 가져오기 위해 추가
from app.schemas.record import RecordResponse
from app.services import ai_engine, alignment

router = APIRouter(prefix="/records", tags=["Records"])

@router.post("/", response_model=RecordResponse, status_code=status.HTTP_201_CREATED)
async def create_record(
    session_id: int = Form(...),
    word_id: int = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. 세션 정보 확인 (어떤 유저의 기록인지 알아내기 위해)
    session_data = db.query(PracticeSession).filter(PracticeSession.session_id == session_id).first()
    if not session_data:
        raise HTTPException(status_code=404, detail="해당 세션을 찾을 수 없습니다.")

    # 2. 목표 단어 정보 가져오기
    word_data = db.query(Word).filter(Word.word_id == word_id).first()
    if not word_data:
        raise HTTPException(status_code=404, detail="단어를 찾을 수 없습니다.")

    # 3. 오디오 파일 읽기
    content = await audio_file.read()

    # 4. [AI Engine] 음성 인식 및 음소 변환
    # ai_result 예시: {"child_text": "사과", "child_phonemes": "ㅅㅏㄱㅗㅏ"}
    ai_result = ai_engine.run_ai_pipeline(content, word_data.word_text)
    
    # 5. [Alignment] 정답과 비교하여 상세 에러 분석
    # errors_list 예시: [{"target": "ㄱ", "error": "ㅂ", "type": "substitution", "pos": "onset"}]
    errors_list = alignment.align_phonemes(
        word_data.standard_phonemes, # 목표 음소
        ai_result["child_phonemes"]   # 아이 음소
    )

    # 6. Record (학습 결과) 저장 - user_id 포함!
    new_record = Record(
        user_id=session_data.user_id,  # [중요] 세션에서 user_id를 가져와 저장
        session_id=session_id,
        word_id=word_id,
        child_text=ai_result["child_text"],
        child_phonemes=ai_result["child_phonemes"],
        is_correct=len(errors_list) == 0,
        created_at=datetime.utcnow()
    )
    db.add(new_record)
    db.flush() # record_id를 미리 생성 (commit 전)

    # 7. Phoneme_Errors (상세 오답) 저장
    error_objects = []
    for err in errors_list:
        new_error = PhonemeError(
            record_id=new_record.record_id,
            target_phoneme=err["target"],
            error_phoneme=err["error"],
            error_type=err["type"],
            error_position=err["pos"]
        )
        db.add(new_error)
        error_objects.append(new_error)
    
    db.commit()
    db.refresh(new_record)

    # 최종 응답 구성
    return {
        "record_id": new_record.record_id,
        "session_id": new_record.session_id,
        "word_id": new_record.word_id,
        "is_correct": new_record.is_correct,
        "child_text": new_record.child_text or "분석 실패",
        "child_phonemes": new_record.child_phonemes or "",
        "created_at": new_record.created_at,
        "errors": error_objects
    }