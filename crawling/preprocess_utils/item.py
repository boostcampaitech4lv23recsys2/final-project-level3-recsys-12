import pandas as pd
import numpy as np
import re
from preprocess_utils.utils_category import *
from preprocess_utils.utils_similarity import *
from preprocess_utils.utils_cluster import *
import pickle
from tqdm import tqdm, tqdm_pandas
tqdm.pandas()


class Item:
    def __init__(self, df, args):
        self.item = df
        self.args = args
        self.item = self.preprocessing()
    
    def preprocessing(self):
        # self.process_price() # price 결측값 total_price로 채우기
        self.process_predict_price()    # predict_price에서 예상 가격만 남기기
        self.process_rating()   # 별점 숫자만 남기기
        self.process_review()   # 리뷰 숫자만 남기기
        self.process_category() # 카테고리 앞뒤에 특수문자(") 제거
        self.process_title()
        self.add_similarity()
        self.add_cluster()
        
        return self.item
    
    def process_price(self):
        self.item = self.item.fillna(np.nan)
        self.item.loc[(~self.item.total_price.isna()), "price"] = self.item.total_price
        # price 결측값 85625 -> 69945
        self.item.drop("total_price", axis=1, inplace=True)
    
    def process_predict_price(self):
        predict_price_list = []

        p = re.compile('.+(?=사)')
        encoder = {idx:i for i, idx in enumerate(self.item.predict_price.index)}
        for i in self.item.predict_price.index:
            m = p.findall(str(self.item.predict_price.iloc[encoder[i]]))
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
    
    def process_title(self):
        self.item["original_title"] = self.item.title
        self.item.rename(columns={"title":"preprocessed_title"}, inplace=True)
        self.item = df_lower(self.item)
        del_list = get_del_words(self.item)
        self.item = df_del_word(self.item, del_list)
        self.item = df_lower(self.item)
        self.item = df_strip(self.item)
        self.item = df_strip2(self.item)
        self.item.preprocessed_title = self.item.preprocessed_title.str.replace(pat=r'[^\w]',repl=r' ',regex=True)
        self.item.preprocessed_title.fillna("ETC", inplace=True)
        print(f"주의: item 15만에 토크나이저 작용하는데 약 8분 정도 소요됩니다!")
        self.item.preprocessed_title = self.item.preprocessed_title.progress_apply(lambda x:tokenize(x))
        print(f"###########################    item preprocessing: PHASE_1_basic_preprocess    ###########################")
        self.item.to_csv(self.args.output_path + f"PHASE_1_basic_preprocess_item.tsv", sep="\t", index=False)
        
    def add_similarity(self):
        self.item.preprocessed_title.fillna("etc", inplace=True)
        self.item.preprocessed_title = self.item.preprocessed_title.str.lower()
        self.item.sort_values("preprocessed_title", inplace=True)
        
        print(f"주의: item 15만 & window_size 100기준으로 약 30분 소요됩니다!")
        
        data, sim_list, sim_list2 = get_similarity_list(self.item, self.args)
        self.item["similarity_list"] = data
        print(f"###########################    item preprocessing: PHASE_2_add_similarity    ###########################")
        self.item.to_csv(self.args.output_path + f"PHASE_2_add_similarity_item.tsv", sep="\t", index=False)
    
    def add_cluster(self):
        del_list = get_del_list()
        self.item.category.fillna("", inplace=True)
        if self.args.test:
            with open(self.args.output_path + "similarity.pickle", "rb") as pkl:
                data = pickle.load(pkl)
        else:
            with open(self.args.similarity_list_path, "rb") as pkl:
                data = pickle.load(pkl)
        self.item["similarity_list"] = data
        self.item = self.item[(~self.item.category.str.contains("|".join(del_list))) & ~(self.item.category=="")]
        self.item.sort_values("item", inplace=True)

        # TODO clustering_threshold 이상의 similarity 가지는 아이템을 저장하는 list 칼럼 생성
        def get_item(x, clustering_threshold):
            tmp = [a[2] for a in x if a[3] >= clustering_threshold]
            return tmp
        self.item["similar_dflist"] = self.item.similarity_list.apply(lambda x:get_item(x, self.args.clustering_threshold))
        self.item = get_cluster_by_BFS(self.item, self.args)
        self.item.drop(columns=["similar_dflist"], inplace=True)
        print(f"###########################    item preprocessing: PHASE_3_add_cluster    ###########################")
        self.item.to_csv(self.args.output_path + f"PHASE_3_add_cluster_item.tsv", sep="\t", index=False)