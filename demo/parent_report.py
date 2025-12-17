import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "report_prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    report_template = f.read()


def generate_parent_report_from_log(log_data, model="gpt-3.5-turbo"):
    log_json_str = json.dumps(log_data, ensure_ascii=False)
    prompt_text = report_template.format(LOG_DATA=log_json_str)

    response = openai.chat.completions.create(  
        model=model,
        messages=[
            {"role": "system", "content": "당신은 아동 발음 학습 전문가입니다."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.7,
        max_tokens=800
    )

    report_text = response.choices[0].message.content.strip()
    return report_text


if __name__ == "__main__":
    sample_log = [
        {"word": "사과", "attempts": 5, "correct": 3, "date": "2025-12-01", "errors": [{"phoneme":"ㅅ","position":"초성","type":"교체"}]},
        {"word": "바나나", "attempts": 4, "correct": 2, "date": "2025-12-02", "errors": [{"phoneme":"ㄴ","position":"초성","type":"탈락"}]}
    ]

    report = generate_parent_report_from_log(sample_log)
    print("✅ 부모 보고서 및 격려 문구 예시:")
    print(report)
