from openai import OpenAI
from services.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def correct_text(input_text):
    """GPT-4o-mini를 이용하여 텍스트 교정"""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
            {"role": "system", "content": "You are an expert in Korean text correction and Chinese interpretation. " \
                                          "Correct the speech of a Korean patient in terms of spelling and context, " \
                                          "and change informal language into polite formal language for communication between a Korean patient and a Chinese caregiver. " \
                                          "If the text ends with the word 'Chinese', only provide the corrected speech in Chinese."},
            {"role": "user", "content": f"Correct the following sentence.: {input_text}"},
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
