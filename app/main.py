from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.session import engine
from app.models.base import Base
from app.models import user, word, step, step_word, practice_session, record, phoneme_error
from app.routers import users, steps, sessions, records
from app.services.ai_engine import load_model

# 서버 시작 및 종료 시 실행될 로직
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. DB 테이블 생성
    Base.metadata.create_all(bind=engine)
    # 2. AI 모델 로드
    await load_model()
    yield

app = FastAPI(lifespan=lifespan)

# 3. CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 라우터 등록
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(steps.router, prefix="/steps", tags=["Steps"])
app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
app.include_router(records.router, prefix="/records", tags=["Records"])

@app.get("/")
def root():
    return {"message": "SsukSsuk Server is running"}
