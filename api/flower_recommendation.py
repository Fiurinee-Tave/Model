import numpy as np
import pandas as pd
import json
import re

from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

#데이터 로드
df_nlp = pd.read_csv("C:/Users/pc/Desktop/민지/동아리/프로젝트(13기)/추천시스템_데이터.csv")

# JSON 문자열을 배열로 변환
df_nlp['최종_벡터'] = df_nlp['최종_벡터'].apply(lambda x: np.array(json.loads(x)))
df_nlp.head(5)


#특수문자 및 띄어쓰기 제거
def remove_special_characters(text):
    # 특수문자 제거
    pattern = r'[^\w\s\.]' #문자,공백문자,마침표 제외 제거
    clean_text = re.sub(pattern, '', text)

    # 한자 제거
    pattern = r'[\u4e00-\u9fff]' #중국어 한자의 유니코드 시작과 끝 제거
    clean_text = re.sub(pattern, '', clean_text)
    clean_text = ' '.join(clean_text.split())
    return clean_text


# 전역 변수 및 모델 초기화
model_name = 'bert-base-multilingual-cased' 
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 문장을 벡터화 변환하는 함수
def get_sentence_embedding(text, tokenizer, model ): 
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True) #최대길이 초과 시 잘라내기, 작은 경우 패딩진행
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy() #각 토큰 벡터의 첫 번째 벡터 확인 (CLS 토큰 벡터)


# 사용자 입력 텍스트 분석 함수(월, 계절)
def extract_month_season(text):
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


# 원핫 벡터 생성
encoder = OneHotEncoder()
encoder.fit_transform(df_nlp[['월', '계절']]).toarray()  # '월'과 '계절' 컬럼을 원핫인코딩

#사용자 입력을 최종 벡터화 
def get_user_input_vector(user_input):
    input = remove_special_characters(user_input) #특수문자 제거
    
    # 토큰화 및 벡터화
    input_embeddings = get_sentence_embedding(input, tokenizer, model) #(1,768) #
    month, season = extract_month_season(user_input)
    user_onehot_vector = np.zeros(len(encoder.get_feature_names_out(['월', '계절'])))
    if month is not None:
        month_idx = encoder.get_feature_names_out(['월', '계절']).tolist().index(f'월_{month}')
        user_onehot_vector[month_idx] = 1
    if season is not None:
        season_idx = encoder.get_feature_names_out(['월', '계절']).tolist().index(f'계절_{season}')
        user_onehot_vector[season_idx] = 1
    user_onehot_vector = user_onehot_vector.reshape(1, 16) #(1,16)

    #벡터 결합
    user_vector = np.concatenate((input_embeddings, user_onehot_vector), axis=1) #(1,784)
    return user_vector


# 추천 시스템 함수 (코사인 유사도 기반)
def recommend_flower(user_input, df, tokenizer, model):
    user_vector = get_user_input_vector(user_input) #사용자 입력을 최종 벡터화

    #코사인 유사도 산출
    df['유사도'] = df['최종_벡터'].apply(lambda x: cosine_similarity(user_vector, x)[0][0])
    df['유사도'] = df['유사도'].astype(float) #숫자형으로 변환

    # 유사도를 기준으로 상위 3개의 꽃을 선택하고 중복된 꽃을 제거
    top3 = df.nlargest(3, '유사도').drop_duplicates(subset='꽃')

    # 만약 중복 제거 후 3개의 꽃이 되지 않는 경우, 다시 nlargest로 채우기
    if top3.shape[0] < 3:
        additional_top = df.nlargest(20, '유사도')  # 상위 10개 정도를 선택
        additional_top = additional_top[~additional_top['꽃'].isin(top3['꽃'])]
        top3 = pd.concat([top3, additional_top]).nlargest(3, '유사도').drop_duplicates(subset='꽃')

    return top3[['꽃', '꽃말', '유사도']]


# 추천 함수 수정
def get_recommendations(user_input):
    recommendations = recommend_flower(user_input, df_nlp, tokenizer, model)
    return recommendations[['꽃', '꽃말', '유사도']].to_dict('records')

