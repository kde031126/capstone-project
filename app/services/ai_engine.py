import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
import asyncio

# 전역 변수로 모델 관리
processor = None
model = None

async def load_model():
    global processor, model
    print("--- [임시] 가짜 AI 모델 모드로 시작합니다 (파일 로드 건너뜀) ---")
    
    # 실제 로드 로직은 전부 주석 처리하거나 지웁니다.
    # processor = Wav2Vec2Processor.from_pretrained(...)
    # model = Wav2Vec2ForCTC.from_pretrained(...)
    
    processor = "MockProcessor" # 가짜 객체
    model = "MockModel"         # 가짜 객체

    await asyncio.sleep(0.1)
    print("--- [임시] 가짜 모델 세팅 완료! 서버를 시작합니다. ---")

def get_phonemes_from_audio(audio_path: str):
    """음성 파일을 분석하여 음소열 반환"""
    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    # 결과가 'ㅅㅏㄱㅘ' 처럼 나오도록 처리 (모델 출력 방식에 따라 조정 필요)
    return transcription.replace(" ", "")

def align_phonemes(target: str, pred: str):
    """
    편집 거리(Levenshtein) 기반 Alignment 로직
    Colab의 핵심: target(정답)과 pred(아이 발음)를 비교해 오답 유형 추출
    """
    n, m = len(target), len(pred)
    dp = np.zeros((n + 1, m + 1))

    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    # 역추적(Backtracking)하여 에러 유형 파악
    errors = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and target[i-1] == pred[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1: # Substitution (대치)
            errors.append({
                "target": target[i-1],
                "error": pred[j-1],
                "type": "substitution",
                "position": i - 1
            })
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1): # Deletion (탈락)
            errors.append({
                "target": target[i-1],
                "error": None,
                "type": "deletion",
                "position": i - 1
            })
            i -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1): # Insertion (첨가)
            errors.append({
                "target": None,
                "error": pred[j-1],
                "type": "insertion",
                "position": i # 삽입 위치
            })
            j -= 1
            
    return errors[::-1] # 역순 정렬하여 원래 순서대로 반환