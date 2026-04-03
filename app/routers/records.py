from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import uuid

from app.db.session import get_db
from app.models.record import Record
from app.models.phoneme_error import PhonemeError
from app.schemas.record import RecordResponse
from app.services import ai_engine # 우리가 만든 가짜 엔진!

router = APIRouter()

@router.post("/", response_model=RecordResponse)
async def create_record(
    session_id: int = Form(...),
    word_id: int = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    아이의 음성을 받아 AI로 분석하고 오답 노트를 생성합니다.
    """
    # 1. 파일 저장 로직 (실제 서비스 시 S3나 로컬 폴더에 저장)
    # 지금은 파일이 잘 왔는지만 확인합니다.
    content = await audio_file.read()
    print(f"🎤 분석 시작: 세션 {session_id}, 단어 {word_id}, 파일명 {audio_file.filename}")

    # 2. AI 엔진 호출 (가짜 데이터 반환 중)
    # 나중에 실제 파인튜닝 모델이 완성되면 여기서 진짜 분석을 수행합니다.
    ai_result = ai_engine.predict_speech(content)
    
    # 3. Record (녹음 결과) 저장
    new_record = Record(
        session_id=session_id,
        word_id=word_id,
        child_text="사과", # AI가 인식한 텍스트 (가짜)
        child_phonemes="ㅅ ㅏ ㄱ ㅗ ㅏ", # AI가 분석한 음소 (가짜)
        is_correct=False # 테스트를 위해 틀린 것으로 설정
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    # 4. Phoneme_Errors (음소 단위 오답) 저장
    # AI가 분석한 '틀린 음소' 리스트를 순회하며 저장합니다.
    fake_errors = [
        {"target": "ㅜ", "error": "ㅗ", "type": "substitution", "pos": 4}
    ]
    
    error_objects = []
    for err in fake_errors:
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

    # 5. 최종 응답 (스키마에 맞춰서 리턴)
    return {
        "record_id": new_record.record_id,
        "is_correct": new_record.is_correct,
        "child_text": new_record.child_text,
        "child_phonemes": new_record.child_phonemes,
        "created_at": new_record.created_at,
        "errors": error_objects # 관계형 데이터를 리스트로 묶어 전달
    }