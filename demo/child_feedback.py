# child_feedback.py

import os
import pygame
from gtts import gTTS

def generate_tts(text, lang="ko", save_dir="audio"):
    """
    텍스트를 음성으로 변환하고 재생합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 단어별 고유 파일명 사용
    safe_text = text.replace(" ", "_")
    file_path = os.path.join(save_dir, f"{safe_text}_feedback.mp3")
   
    # TTS 생성
    tts = gTTS(text=text, lang=lang)
    tts.save(file_path)
   
    # pygame으로 재생
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # 재생 끝날 때까지 대기
        pygame.time.Clock().tick(10)
    
    # 재생 종료 후 pygame 종료
    pygame.mixer.music.stop()
    pygame.mixer.quit()
   
    return file_path

def play_standard_pronunciation(target_word):
    """
    아동에게 표준 발음을 들려주는 함수.
    """
    print(f"🔊 '{target_word}'의 표준 발음을 재생합니다.")
    audio_file = generate_tts(target_word)
    print(f"✅ 재생 완료: {audio_file}")
    return audio_file

# 테스트용
if __name__ == "__main__":
    play_standard_pronunciation("사과")
    play_standard_pronunciation("바나나")