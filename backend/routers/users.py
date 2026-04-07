# app/routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.session import get_db
from models.user import User
from schemas.user import UserCreate, UserResponse

router = APIRouter() # main.py에서 prefix를 붙이므로 여기선 비워둬도 됩니다.

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_or_login_user(user: UserCreate, db: Session = Depends(get_db)):
    # 1. 기존 유저인지 확인 (Firebase UID 기준)
    # 팁: .filter() 안에 정확한 컬럼명을 확인하세요!
    db_user = db.query(User).filter(User.firebase_uid == user.firebase_uid).first()
    
    # 2. 이미 존재하는 유저라면 새로 만들지 않고 기존 정보를 바로 반환 (로그인 처리)
    if db_user:
        print(f"기존 유저 발견: {db_user.user_name}")
        return db_user
    
    # 3. 신규 유저 저장 (회원가입 처리)
    try:
        # Pydantic v2에서는 .dict() 대신 .model_dump()를 씁니다. 아주 잘하셨어요!
        new_user = User(**user.model_dump()) 
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"신규 유저 등록 완료: {new_user.user_name}")
        return new_user
    except Exception as e:
        db.rollback() # 에러 발생 시 DB 상태를 되돌립니다.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"유저 등록 중 오류가 발생했습니다: {str(e)}"
        )

# 유저 정보 조회 API
@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다.")
    return db_user