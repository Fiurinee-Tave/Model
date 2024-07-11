import pandas as pd
import numpy as np
import re

class Flower_Color_List:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
    
    def result_flower(self, flower_name, flower_mean):
        result_flower=self.df[(self.df['꽃']==flower_name) & (self.df['꽃말']==flower_mean)]

        combined_result = pd.concat([
        result_flower[['꽃1', '선택한 색상1']].rename(columns={'꽃1': '꽃', '선택한 색상1': '선택한 색상'}),
        result_flower[['꽃2', '선택한 색상2']].rename(columns={'꽃2': '꽃', '선택한 색상2': '선택한 색상'})
        ], axis=0).reset_index(drop=True)

        return combined_result.to_dict(orient='records')


# 인스턴스 초기화
recommender = Flower_Color_List(data_path="dataset/어울리는색상_데이터.csv")