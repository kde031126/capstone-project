# run_asr_inference.py

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
from pydub import AudioSegment
import hgtk
import re
from g2pk import G2p
g2p = G2p()

# warning 문구들 지우기
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ffmpeg / ffprobe 경로 수동 지정
os.environ["FFMPEG_BINARY"] = r"C:\FFmpeg\bin\ffmpeg.exe"
os.environ["FFPROBE_BINARY"] = r"C:\FFmpeg\bin\ffprobe.exe"
AudioSegment.converter = r"C:\FFmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\FFmpeg\bin\ffprobe.exe"

# mp3 → wav 변환 함수
def ensure_wav(audio_path: str, target_sampling_rate: int = 16000):
    """
    mp3 파일을 wav로 변환하고, wav 파일 경로 반환.
    """
    if audio_path.endswith(".mp3"):
        wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(target_sampling_rate).set_channels(1)
        audio.export(wav_path, format="wav")
        return wav_path
    return audio_path

# 1. 모델 설정
MODEL_ID = "kresnik/wav2vec2-large-xlsr-korean" 

# 2. 모델 로드
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    print(f"✅ 모델 '{MODEL_ID}' 로드 완료.")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()

# 3. 오디오 파일 → numpy 배열
def speech_file_to_array_fn(path: str, target_sampling_rate: int = 16000):
    try:
        speech_array, sampling_rate = torchaudio.load(path)
        if sampling_rate != target_sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=target_sampling_rate
            )
            speech_array = resampler(speech_array)
        return speech_array.squeeze().numpy()
    except Exception as e:
        print(f"Error loading file {path}: {e}")
        return None

# 4. ASR 추론
def transcribe_audio(audio_path: str):
    print(f"\n🔬 분석 시작: {audio_path}")
    audio_input = speech_file_to_array_fn(audio_path)
    if audio_input is None:
        print("오디오 로드 실패 - None 반환")
        return []
    
    input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"ASR 원문: {transcription}")

    child_phoneme_sequence = []
    try:
        # 0. ASR 결과에서 불필요한 특수문자 제거
        temp_transcription = transcription.replace('ᴥ', '').strip()

        # 1. 한글 음절 이외의 모든 문자 제거
        clean_text = re.sub(r'[^가-힣]', '', temp_transcription)

        # 2. 한글 음절을 초/중/종성으로 분해하여 phoneme_sequence 만들기
        for char in clean_text:
            try:
                # ASR이 인식한 음절을 초/중/종성으로 분해
                cho, jung, jong = hgtk.letter.decompose(char)
                
                # 초성 처리
                if cho != ' ':
                    child_phoneme_sequence.append(f"{cho}_초")
                    
                # 중성 처리
                if jung != ' ':
                    for m in list(jung):
                        child_phoneme_sequence.append(f"{m}_중")
                
                # 종성 처리 (**추가됨**)
                if jong != '':
                    child_phoneme_sequence.append(f"{jong}_종")
                else:
                    # 종성이 없는 경우 명시적으로 ∅_종 추가
                    child_phoneme_sequence.append("∅_종")
                    
            except hgtk.exception.NotHangulException:
                continue
                      
    except Exception as e:
        print(f"경고: hgtk 자모 분해 실패. ASR 결과 문자열 그대로 사용: {transcription}, 오류: {e}")
        child_phoneme_sequence = list(re.sub(r'\s+', '', transcription))

    print("------------------------------------------")
    print(f"최종 인식 결과 : {transcription}")
    print(f"최종 인식 결과 (음소열 리스트): {child_phoneme_sequence}") # 리스트 출력 확인
    print("------------------------------------------")
    return child_phoneme_sequence

# 5. 테스트 오디오 파일 경로
TEST_AUDIO_PATH = r"demo\test_audio.mp3"  # 실제 mp3 파일
TEST_AUDIO_PATH = ensure_wav(TEST_AUDIO_PATH)  # mp3 → wav 변환

def get_standard_phonemes_with_position_g2pk(target_text: str):
    """
    타겟 텍스트(한글)를 g2pk로 표준 발음을 변환한 후,
    초/중/종성 위치 태그가 붙은 표준 음소열 리스트로 변환합니다.
    예: '닭이' (발음: '달기') -> ['ㄷ_초', 'ㅏ_중', 'ㄹ_종', 'ㄱ_초', 'ㅣ_중']
    예: '좋아요' (발음: '조아요') -> ['ㅈ_초', 'ㅗ_중', 'ㅇ_초', 'ㅏ_중', '요_중']
    """
    standard_phoneme_sequence = []
    
    # 1. g2pk를 사용하여 타겟 텍스트를 표준 발음 문자열로 변환
    # 이 과정에서 띄어쓰기 및 음운 변동이 반영됩니다.
    # 예: '닭이 먹다' -> '달기 먹따'
    try:
        # NOTE: g2pk는 기본적으로 띄어쓰기를 보존합니다.
        standard_pronunciation_text = g2p(target_text)
    except Exception as e:
        print(f"g2pk 변환 오류: {e}")
        return []

    # 2. 띄어쓰기 및 특수문자 제거 후 순수한 한글 음절만 남김
    # g2pk 결과는 특수문자(예: !?)나 영어는 그대로 남기므로, 한글 음절만 필터링합니다.
    clean_text = re.sub(r'[^가-힣]', '', standard_pronunciation_text)
    
    for char in clean_text:
        try:
            # hgtk.letter.decompose: '과' -> ('ㄱ', 'ㅗ', 'ㅏ') 
            cho, jung, jong = hgtk.letter.decompose(char)
            
            # 표준 발음이 적용된 후의 초/중/종성 분해
            
            # 초성 처리
            if cho != ' ': # 초성이 있는 경우
                standard_phoneme_sequence.append(f"{cho}_초")
                
            # 중성 처리 (복수 중성, 즉 이중모음 포함)
            # hgtk 분해 결과가 3자리이므로, 중성(jung)은 모음 하나 또는 두 개(이중모음)를 포함
            if jung != ' ':
                # 복수 중성을 분리하여 처리
                # 예: 'ㅘ'는 'ㅗ'와 'ㅏ'로 분리되어 각각 'ㅗ_중', 'ㅏ_중'으로 처리
                for m in list(jung):
                    standard_phoneme_sequence.append(f"{m}_중")
            
            # 종성 처리
            if jong != '':
                standard_phoneme_sequence.append(f"{jong}_종")
            else:
                # 종성이 없는 경우 명시적으로 ∅_종 추가
                standard_phoneme_sequence.append("∅_종")
                
        except hgtk.exception.NotHangulException:
            # 한글이 아닌 문자
            continue
            
    return standard_phoneme_sequence

# 2. Sequence Alignment (Levenshtein DP + 역추적)
# ----------------------------------------------------
def perform_sequence_alignment_levenshtein(standard_seq, child_seq):
    """
    표준 음소열과 아동 음소열을 비교하여 오류 라벨을 생성합니다.
    Levenshtein 거리 기반 DP + 역추적으로 정확한 위치 탐지
    """
    len_s = len(standard_seq)
    len_c = len(child_seq)
    
    # 1. DP 테이블 초기화
    dp = [[0]*(len_c+1) for _ in range(len_s+1)]
    for i in range(len_s+1):
        dp[i][0] = i  # 표준에서 삭제
    for j in range(len_c+1):
        dp[0][j] = j  # 아동에서 삽입
    
    # 2. DP 테이블 채우기
    for i in range(1, len_s+1):
        for j in range(1, len_c+1):
            if standard_seq[i-1] == child_seq[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j-1] + 1,  # 교체
                    dp[i][j-1] + 1,    # 삽입
                    dp[i-1][j] + 1     # 삭제
                )
    
    # 3. 역추적
    i, j = len_s, len_c
    alignment_result = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and standard_seq[i-1] == child_seq[j-1]:
            alignment_result.append({
                'standard_phoneme': standard_seq[i-1],
                'child_phoneme': child_seq[j-1],
                'label': '정확'
            })
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment_result.append({
                'standard_phoneme': standard_seq[i-1],
                'child_phoneme': child_seq[j-1],
                'label': '교체'
            })
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment_result.append({
                'standard_phoneme': '∅',
                'child_phoneme': child_seq[j-1],
                'label': '첨가'
            })
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment_result.append({
                'standard_phoneme': standard_seq[i-1],
                'child_phoneme': '∅',
                'label': '탈락'
            })
            i -= 1
    
    alignment_result.reverse()  # 역순으로 append했으므로 뒤집기
    return alignment_result


def split_phoneme(ph):
    """
    입력: 'ㄱ_초' 또는 '∅'
    출력: phoneme='ㄱ', position='초'
    """
    if ph == '∅':
        return '∅', '없음'
    if '_' not in ph:
        return ph, '미정'
    phoneme, position = ph.split('_')
    return phoneme, position

if __name__ == "__main__":
    print("Wav2Vec 2.0 ASR 추론 및 발음 오류 분석 스크립트")
    
    # 테스트를 위한 타겟 텍스트 정의 
    # 예를 들어, 아동이 "사과"라는 단어를 발음해야 한다고 가정
    TARGET_TEXT = "사과"
    
    print(f"\n--- 타겟 텍스트: '{TARGET_TEXT}' ---")

    # 1. 표준 발음 음소열 생성
    standard_phonemes = get_standard_phonemes_with_position_g2pk(TARGET_TEXT)
    print(f"표준 음소열 (위치 포함): {standard_phonemes}")
    
    # 2. 아동 발음 ASR 추론 및 음소열 변환
    child_phonemes = transcribe_audio(TEST_AUDIO_PATH)
    
    if child_phonemes:
        # 3. Sequence Alignment를 통한 오류 분석
        print("\n--- Sequence Alignment 분석 시작 ---")
        alignment_results = perform_sequence_alignment_levenshtein(standard_phonemes, child_phonemes)
        
        # 4. 분석 결과 출력 (표 형식)
        print("\n[ 최종 발음 오류 분석 결과 ]")
        print("------------------------------------------------------------------------")
        print("{:<15} {:<15} {:<10}".format("표준 음소", "아동 발음", "오류 라벨"))
        print("------------------------------------------------------------------------")
        
        errors = []
        error_count = 0
        for result in alignment_results:
            std_ph = result['standard_phoneme']
            child_ph = result['child_phoneme']
            label = result['label']

            print("{:<15} {:<15} {:<10}".format(
                std_ph, child_ph, label
            ))

            if label != '정확':
                error_count += 1
        
                # 위치 및 음소 분리
                phoneme, position = split_phoneme(child_ph if child_ph != '∅' else std_ph)

                # errors 리스트에 저장
                errors.append({
                    "standard_phoneme": std_ph,
                    "child_phoneme": child_ph,
                    "position": position,
                    "type": label
                })
                
        print("------------------------------------------------------------------------")
        print(f"총 오류 개수: {error_count}")
        for e in errors:
            print(f"- 표준 음소: {e['standard_phoneme']}, 아동 음소: {e['child_phoneme']}, "
          f"위치: {e['position']}, 유형: {e['type']}")
        
    print("\n--- 발음 오류 분석 테스트 완료 ---")