import openai
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

class Flower_Ment:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        openai.api_key = self.api_key
    
    def get_completion(self, prompt, model="gpt-4"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message['content']

    def result_ment(self, user_input, flower_name, flower_mean):
        prompt = (
            f"너는 지금부터 감성적인 시인이며, 대답은 마치 시인이 선물하는 것처럼 해야해. "
            f"꽃과 꽃말과 사용자 입력에 따라서 가장 어울리는 꽃다발 선물 멘트를 한줄로 생성할거야. "
            f"꽃은 반드시 {flower_name}이고 꽃말은 {flower_mean} 이야. 사용자 입력은 {user_input}이야. "
            f"멘트에 꽃과 꽃말 그리고 사용자 입력이 들어가면서 사용자 입장에서의 멘트를 구체적으로 추천해줘. "
            f"반드시 사용자 입장에서의 멘트만 간결하게 70자 이내로 출력해야해"
        )

        return self.get_completion(prompt)