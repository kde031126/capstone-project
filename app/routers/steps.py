from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.models.step import Step
from app.schemas.step import StepResponse

router = APIRouter()

# 1. 전체 단계 목록 조회 (단어 리스트 포함)
@router.get("/", response_model=List[StepResponse])
def get_all_steps(db: Session = Depends(get_db)):
    """
    모든 학습 단계와 각 단계에 포함된 단어 리스트를 반환합니다.
    """
    # DB에서 Step들을 순서(step_order)대로 가져옵니다.
    steps = db.query(Step).order_by(Step.step_order).all()
    
    if not steps:
        # 데이터가 없을 경우 빈 리스트를 반환하거나 404를 띄울 수 있습니다.
        return []
        
    return steps

# 2. 특정 단계 상세 조회
@router.get("/{step_id}", response_model=StepResponse)
def get_step_detail(step_id: int, db: Session = Depends(get_db)):
    step = db.query(Step).filter(Step.step_id == step_id).first()
    if not step:
        raise HTTPException(status_code=404, detail="해당 단계를 찾을 수 없습니다.")
    return step