import pandas as pd
import numpy as np
import re

class Item:
    def __init__(self, df):
        self.item = df
    
    def preprocessing(self):
        self.proces_price() # price 결측값 total_price로 채우기
        self.process_predict_price()    # predict_price에서 예상 가격만 남기기
        self.process_rating()   # 별점 숫자만 남기기
        self.process_review()   # 리뷰 숫자만 남기기
        self.process_category() # 카테고리 앞뒤에 특수문자(") 제거
        
        return self.item
    
    def proces_price(self):
        self.item = self.item.fillna(np.nan)
        self.item.loc[(~self.item.total_price.isna()), "price"] = self.item.total_price
        # price 결측값 85625 -> 69945
        self.item.drop("total_price", axis=1, inplace=True)
    
    def process_predict_price(self):
        predict_price_list = []

        p = re.compile('.+(?=사)')
        for i in self.item.predict_price.index:
            m = p.findall(str(self.item.predict_price.iloc[i]))
            if len(m) == 0:
                m.append("예상가정보없음")
            predict_price_list.append(m[0][3:])
        self.item.predict_price = predict_price_list
    
    def process_rating(self):
        self.item.rating = self.item.rating.apply(lambda x: str(x)[3:-1])
        self.item.rating = self.item.rating.apply(lambda x: np.nan if x == '' else x)
    
    def process_review(self):
        self.item.review = self.item.review.fillna("리뷰 쓰기첫 리뷰 두 배 적립")
        review_list = []

        p = re.compile('[(][0-9]*[)]')
        for review in self.item.review:
            m = p.findall(review)
            if len(m) == 0:
                m.append("(0)")
            review_list.append(m[0][1:-1])

        self.item.review = review_list
    
    def process_category(self):
        self.item.category.fillna("기타", inplace=True)
        self.item.category = self.item.category.apply(lambda x: x.strip('"'))