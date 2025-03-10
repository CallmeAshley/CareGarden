from openai import OpenAI
from services.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def correct_text(input_text):
    """GPT-4o-mini를 이용하여 텍스트 교정"""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
            {"role": "system", "content": "너는 한국어 텍스트 교정 및 중국어 통역 전문가야. "
                                          "한국인 환자와 중국인 간병인 사이 환자의 발화를 맞춤법과 문맥에 맞게 교정하고, 반말을 존댓말로 공손하게 변경해."
                                          "만약 텍스트에 '중국어'라는 단어가 마지막에 포함되어 있으면 중국어로 교정된 발화만 제공해."},
            {"role": "user", "content": f"다음 문장을 교정해: {input_text}"},
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
