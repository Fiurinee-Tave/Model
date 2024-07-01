import numpy as np
import pandas as pd
import json
import re

from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

class FlowerRecommender:
    def __init__(self, data_path):
        self.df_nlp = pd.read_csv(data_path)
        self.df_nlp['최종_벡터'] = self.df_nlp['최종_벡터'].apply(lambda x: np.array(json.loads(x)))
        
        model_name = 'bert-base-multilingual-cased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.encoder = OneHotEncoder()
        self.encoder.fit_transform(self.df_nlp[['월', '계절']]).toarray()  # '월'과 '계절' 컬럼을 원핫인코딩
    
    def remove_special_characters(self, text):
        pattern = r'[^\w\s\.]' #특수문자 제거
        clean_text = re.sub(pattern, '', text)
        pattern = r'[\u4e00-\u9fff]' #한자 제거
        clean_text = re.sub(pattern, '', clean_text)
        clean_text = ' '.join(clean_text.split())
        return clean_text
    
    def get_sentence_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()
    
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

    def recommend(self, user_input):
        #사용자 벡터화
        user_input_cleaned = self.remove_special_characters(user_input)
        user_vector = self.get_sentence_embedding(user_input_cleaned)

        user_onehot_vector = np.zeros(len(self.encoder.get_feature_names_out(['월', '계절'])))
        month, season = self.extract_month_season(user_input)
        if month is not None:
            month_idx = self.encoder.get_feature_names_out(['월', '계절']).tolist().index(f'월_{month}')
            user_onehot_vector[month_idx] = 1
        if season is not None:
            season_idx = self.encoder.get_feature_names_out(['월', '계절']).tolist().index(f'계절_{season}')
            user_onehot_vector[season_idx] = 1
        user_onehot_vector = user_onehot_vector.reshape(1, 16) #(1,16)

        user_result = np.concatenate((user_vector, user_onehot_vector), axis=1) #최종 결합 벡터(1,784)

        #코사인 유사도 산출
        sim_scores = self.df_nlp['최종_벡터'].apply(lambda x: cosine_similarity(user_result, x)[0][0]) 
        top3 = sim_scores.nlargest(3)
        recommendations = self.df_nlp.iloc[top3.index]
        recommendations['유사도'] = top3
        return recommendations[['꽃', '꽃말', '유사도']].to_dict(orient='records')

# 인스턴스 초기화
recommender = FlowerRecommender(data_path="C:/Users/pc/Desktop/민지/동아리/프로젝트(13기)/추천시스템_데이터.csv")