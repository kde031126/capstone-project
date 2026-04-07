import torch
import librosa
from g2pk import G2p
from jamo import hangul_to_jamo, jamo_to_hcj
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io

# 1. 전역 변수 및 모델 로드 세팅
g2p = G2p()
processor = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def load_model():
    """서버 시작 시 모델을 메모리에 로드 (비동기 처리)"""
    global processor, model
    print(f"--- [AI] 모델 로딩 시작 (Device: {device}) ---")
    
    # 코랩 예시의 'korean-wav2vec2-phoneme' 모델 기준
    model_name = "kresnik/wav2vec2-large-xlsr-korean" # 예시 모델명
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    
    print("--- [AI] Wav2Vec 2.0 & g2pK 로드 완료 ---")

# --- [Pipeline Step 1: 표준 음소 추출] ---
def get_standard_phonemes(text: str) -> str:
    """텍스트를 받아서 g2pK로 표준 발음 음소열 생성"""
    # 1. g2p로 발음 교정 (예: '국물' -> '궁물')
    pronounced_text = g2p(text)
    # 2. 자모 분리 (예: '궁물' -> 'ㄱㅜㅇㅁㅜㄹ')
    jamo_str = hangul_to_jamo(pronounced_text)
    # 3. 현대 한글 자모로 변환 및 공백으로 구분
    phonemes = " ".join(list(jamo_to_hcj(jamo_str)))
    return phonemes

# --- [Pipeline Step 2: 아동 발화 음소 추출] ---
def predict_child_phonemes(audio_bytes: bytes) -> str:
    """음성 데이터를 받아서 Wav2Vec 2.0으로 인식된 음소열 생성"""
    # 1. 오디오 전처리 (16kHz 변환)
    audio_file = io.BytesIO(audio_bytes)
    speech, _ = librosa.load(audio_file, sr=16000)
    
    # 2. 모델 추론
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    
    # 3. CTC Decoding (가장 확률 높은 음소 선택)
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # 4. 결과 정제 (공백 추가 등)
    jamo_str = hangul_to_jamo(transcription.replace(" ", "")) # 공백 제거 후 분리
    clean_phonemes = " ".join(list(jamo_to_hcj(jamo_str)))
    
    return transcription, clean_phonemes

# --- [통합 엔트리 포인트] ---
def run_ai_pipeline(audio_content: bytes, target_text: str):
    """표준 음소와 인식 음소를 동시에 추출하여 반환"""
    # 표준 정답 생성
    standard = get_standard_phonemes(target_text)
    
    # 2. 아동 발화 인식 (언패킹으로 두 값을 받음)
    recognized_text, recognized_phonemes = predict_child_phonemes(audio_content)
    
    return {
        "target_phonemes": standard,
        "child_text": recognized_text,
        "child_phonemes": recognized_phonemes
    }