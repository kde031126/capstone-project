from fastapi import FastAPI
from app.db.session import engine
from app.models.base import Base
import app.models.user
import app.models.word
import app.models.step
import app.models.step_word
import app.models.practice_session
import app.models.record
import app.models.phoneme_error
from app.routers import users

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(users.router)


@app.get("/")
def root():
    return {"message": "Server is running"}
