import pandas as pd
import numpy as np
import re

class Flower_Color_List:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
    
    def result_flower(self, user_index):
        self.df[['꽃1', '꽃2']].iloc[0,:]


# 인스턴스 초기화
recommender = Flower_Color_List(data_path="dataset/어울리는색상_데이터.csv")