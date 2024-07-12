import numpy as np
import pandas as pd
import json
import re

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OneHotEncoder
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity

class FlowerRecommender:
    def __init__(self, data_path):
        self.df_nlp = pd.read_csv(data_path)
        self.df_nlp['설명_벡터'] = self.df_nlp['설명_벡터'].apply(lambda x: np.array(json.loads(x)))
        self.df_nlp['색상_벡터'] = self.df_nlp['색상_벡터'].apply(lambda x: np.array(json.loads(x)))
        self.df_nlp['원핫인코딩'] = self.df_nlp['원핫인코딩'].apply(lambda x: np.array([float(num) for num in x.strip('[]').split()]).reshape(1, -1))

        model_name = 'klue/roberta-small'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.encoder = OneHotEncoder()
        self.encoder.fit_transform(self.df_nlp[['월', '계절']]).toarray()  # '월'과 '계절' 컬럼을 원핫인코딩

        self.okt = Okt()

        self.color_synonyms = {
            '갈색': ['갈색', '브라운', '갈색의', '브라운색', '갈색이', '갈색이다', '갈색을', '갈색으로'],
            '노랑': ['노랑', '노란', '노란색', '황색', '노랑색', '노랗다', '노랗게', '노란빛', '노란 빛','누런'],
            '보라': ['보라', '보라색', '자주색', '보라빛', '보랏빛', '보라빛의', '보라빛이', '보라빛으로'],
            '분홍': ['분홍', '핑크', '분홍색', '핑크색', '분홍빛', '분홍빛의', '분홍빛이', '분홍빛으로'],
            '빨강': ['빨강', '빨간', '빨강색', '빨간색', '붉은', '붉은색', '붉은 빛', '붉다', '붉게'],
            '주황': ['주황', '주황색', '오렌지', '오렌지색', '주황빛', '주황빛의', '주황빛이', '주황빛으로'],
            '초록': ['초록', '초록색', '녹색', '초록빛', '초록의', '초록이', '초록으로'],
            '파랑': ['파랑', '파란', '파란색', '파랑색', '청색', '파랑빛', '파란빛', '파랗다', '파랗게','푸른','푸른빛', '푸른 색'],
            '흰색': ['흰색', '하양', '하얀', '백색', '하얀색', '하양색', '하얗다', '하얗게', '백색의', '백색이', '백색으로','흰']
            }
    
    def remove_special_characters(self, text):
        if not isinstance(text, str):
            text = str(text)
        pattern = r'[^\w\s\.]' # 문자, 공백문자, 마침표 제외 제거
        clean_text = re.sub(pattern, '', text)

        pattern = r'[\u4e00-\u9fff]' # 중국어 한자의 유니코드 시작과 끝 제거
        clean_text = re.sub(pattern, '', clean_text)
        clean_text = ' '.join(clean_text.split())
        return clean_text
   
    def get_sentence_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True) 
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    #색상 유무에 따라 최종 벡터 다르게 생성
    def create_combined_vector(self, row):
        설명_벡터 = row['설명_벡터']
        색상_벡터 = row['색상_벡터']
        원핫인코딩 = row['원핫인코딩']
        combined_array = np.concatenate((설명_벡터, 색상_벡터, 원핫인코딩), axis=1)
        return combined_array
    
    def create_combined_vector_without_color(self, row):
        설명_벡터 = row['설명_벡터']
        원핫인코딩 = row['원핫인코딩']
        combined_array = np.concatenate((설명_벡터, 원핫인코딩), axis=1)
        return combined_array
    

    def extract_month_season(self, text): # 사용자 입력 텍스트 분석 함수(월, 계절)
        months = {
            '1월': 1, '2월': 2, '3월': 3, '4월': 4, '5월': 5, '6월': 6,
            '7월': 7, '8월': 8, '9월': 9, '10월': 10, '11월': 11, '12월': 12
        }
        seasons = {'봄': '봄', '여름': '여름', '가을': '가을', '겨울': '겨울'}
        month = None
        season = None

        for key, value in months.items():
            if key in text:
                month = value
                break
        for key in seasons.keys():
            if key in text:
                season = key
                break
        return month, season
    
    def extract_color(self, text):
        text = text.lower()
        for color, synonyms in self.color_synonyms.items():
            for synonym in synonyms:
                if synonym in text:
                    return color
        return None

    def is_noun_phrase(self, text):
        pos_tags = self.okt.pos(text)
        for word, tag in pos_tags:
            if tag != 'Noun':
                return False
        return True

    def add_context_if_noun(self, user_input):
        if self.is_noun_phrase(user_input):
            return user_input + "에 어울리는 꽃을 추천해."
        return user_input


    def get_user_input_vector(self, user_input, user_month):
        input_text = self.remove_special_characters(user_input)
        input_embeddings = self.get_sentence_embedding(input_text)
        input_color_text = self.extract_color(user_input)
        month, season = self.extract_month_season(user_input)

        # 원핫 벡터 생성
        user_onehot_vector = np.zeros(len(self.encoder.get_feature_names_out(['월', '계절'])))
        if month is None and season is None: #사용자가 쓰는 날의 달 추출
            month = user_month
            if month in [3, 4, 5]:
                season = '봄'
            elif month in [6, 7, 8]:
                season = '여름'
            elif month in [9, 10, 11]:
                season = '가을'
            else:
                season = '겨울'
            month_idx = self.encoder.get_feature_names_out(['월', '계절']).tolist().index(f'월_{month}')
            season_idx = self.encoder.get_feature_names_out(['월', '계절']).tolist().index(f'계절_{season}')
            user_onehot_vector[month_idx] = 1
            user_onehot_vector[season_idx] = 1

        if season is None and month is not None: #사용자가 입력한 월 기준으로 계절 추출
            if month in [3, 4, 5]:
                season = '봄'
            elif month in [6, 7, 8]:
                season = '여름'
            elif month in [9, 10, 11]:
                season = '가을'
            else:
                season = '겨울'
            season_idx = self.encoder.get_feature_names_out(['월', '계절']).tolist().index(f'계절_{season}')
            user_onehot_vector[season_idx] = 1
            
        if month is not None: #텍스트에 입력된 월
            month_idx = self.encoder.get_feature_names_out(['월', '계절']).tolist().index(f'월_{month}')
            user_onehot_vector[month_idx] = 1
        if season is not None: #텍스트에 입력된 계절
            season_idx = self.encoder.get_feature_names_out(['월', '계절']).tolist().index(f'계절_{season}')
            user_onehot_vector[season_idx] = 1
        user_onehot_vector = user_onehot_vector.reshape(1, -1) #(1,16)

        # 벡터 결합
        if input_color_text:
            color_embeddings = self.get_sentence_embedding(input_color_text)
            user_vector = np.concatenate((input_embeddings, color_embeddings, user_onehot_vector), axis=1)
        else:
            user_vector = np.concatenate((input_embeddings, user_onehot_vector), axis=1)
        return user_vector

    def apply_event_weight_for_row(self, user_input, row):
        event_weights = {
            '발렌타인': 1.2,
            '화이트': 1.2,
            '어버이날': 1.2,
            '어버이': 1.2,
            '부모': 1.2,
            '부모님': 1.2,
            '성년의날': 1.2,
            '성년': 1.2,
            '로즈데이': 1.2,
            '로즈': 1.2,
            '스승의날': 1.2,
            '선생': 1.2,
            '스승': 1.2,
            '선생님': 1.2,
            '연인': 1.2,
            '여자친구': 1.2,
            '남자친구': 1.2,
            '여친': 1.2,
            '남친': 1.2,
            '아내': 1.2,
            '와이프': 1.2,
            '남편': 1.2,
            '생일': 1.2,
            '기념일': 1.2,
            '사랑': 1.2
        }
        weight = 1.0

        for event, event_weight in event_weights.items():
            if event in user_input:
                weight *= event_weight
                break

        #사용자 월 추출
        month, season = self.extract_month_season(user_input)

        if any(event in row['설명'] for event in event_weights.keys() if event in user_input):
            weight *= 1.2
        if month and isinstance(row['월'], int) and month == row['월']:
            weight *= 1.2
        if season and isinstance(row['계절'], str) and season == row['계절']:
            weight *= 1.2
        return weight
    
    def recommend_flower(self, user_input, user_month=None):
        user_input = self.add_context_if_noun(user_input)
        user_vector = self.get_user_input_vector(user_input, user_month)
        input_color = self.extract_color(user_input)
        df = self.df_nlp

        if input_color:
            # 색상이 명시된 경우 해당 색상의 꽃들로 필터링
            filtered_df = df[df['색상'] == input_color]
            filtered_df['최종_벡터'] = filtered_df.apply(lambda row: self.create_combined_vector(row), axis=1)
        else:
            # 색상이 명시되지 않은 경우 결합 벡터에서 색상 벡터를 제외
            filtered_df = df.copy()
            filtered_df['최종_벡터'] = filtered_df.apply(lambda row: self.create_combined_vector_without_color(row), axis=1)

        # 각 행에 대해 가중치 계산 및 유사도 산출
        filtered_df['유사도'] = filtered_df.apply(lambda row: cosine_similarity(user_vector, np.array(row['최종_벡터']).reshape(1, -1))[0][0] * self.apply_event_weight_for_row(user_input, row), axis=1)
        filtered_df['유사도'] = filtered_df['유사도'].astype(float)  # 숫자형으로 변환

        # 유사도를 기준으로 상위 3개의 꽃을 선택하고 중복된 꽃을 제거
        top3 = filtered_df.nlargest(3, '유사도').drop_duplicates(subset='꽃')

        # 만약 중복 제거 후 3개의 꽃이 되지 않는 경우, 다시 nlargest로 채우기
        if top3.shape[0] < 3:
            additional_top = filtered_df.nlargest(20, '유사도')  # 상위 20개 정도를 선택
            additional_top = additional_top[~additional_top['꽃'].isin(top3['꽃'])]
            top3 = pd.concat([top3, additional_top]).nlargest(3, '유사도').drop_duplicates(subset='꽃')

        return top3[['꽃', '꽃말', '유사도']].to_dict('records')
    
# 인스턴스 초기화
recommender = FlowerRecommender(data_path="dataset/추천시스템_데이터.csv")