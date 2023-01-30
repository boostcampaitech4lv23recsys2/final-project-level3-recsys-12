import pandas as pd
from tqdm import tqdm
import argparse
from time import time
import pickle
from utils_cluster import *
from time import time

now = str(round(time()))[5:]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_data_path", default="/opt/ml/input/final/EDA/clustered_item_09305.csv", type=str)
    parser.add_argument("--train_data_path", default="/opt/ml/input/data/hi_interaction.csv", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    item = pd.read_csv(args.item_data_path)
    train = pd.read_csv(args.train_data_path)
    
    # item에 포함되지 않은 hi_interaction item을 제거하기
    train = train[train.item.isin(item.item.unique())]
    dummy = list()
    c = item.cls_id.max()

    for i in range(len(item[item.cls_id == -1])):
        dummy.append(i + c + 1)
    item.loc[item.cls_id == -1, "cls_id"] = dummy

    item2cls = {item_id:cls_id for item_id, cls_id in zip(item.item, item.cls_id)}

    train["cluster_id"] = train.item.map(item2cls)

    train.to_csv("clustered_train.csv", index=False)
    
    predict_price_list = []

    import re
    p = re.compile('.+(?=사)')
    for i in item.predict_price.index:
        m = p.findall(str(item.predict_price.iloc[i]))
        if len(m) == 0:
            m.append("예상가정보없음")
        if m[0][3:] == "정보없음":
            predict_price_list.append("정보없음")
        else:
            predict_price_list.append(int("".join((m[0][3:]).split(","))))

    item.predict_price = predict_price_list

    item.loc[(item.predict_price != "정보없음"), "price"] = item[(item.predict_price != "정보없음")].predict_price

    import re
    def f(x):
        pattern = re.compile(r"별점\s+(\d+\.\d+)점")
        try:
            match = pattern.search(x)
            if match:
                score = float(match.group(1))
            else:
                score = None
            return score
        except:
            return 0.0

    item.rating = item.rating.apply(lambda x: f(x))

    item.review = item.review.fillna("리뷰 쓰기첫 리뷰 두 배 적립")
    review_list = []

    import re
    p = re.compile('[(][0-9]*[)]')
    for idx, review in enumerate(item.review):
        m = p.findall(review)
        if len(m) == 0:
            m.append("(0)")
        review_list.append(m[0][1:-1])

    item.review = review_list

    item.review = item.review.astype(int)
    item.rating = item.rating.astype(float)

    item = item.sort_values(["cls_id", "review", "rating"], ascending=[True, False, False])

    item.groupby("cls_id")["item"].apply(lambda x:len(list(x)))

    cls_major_item_id = item.rename(columns={"item":"major_item"}).groupby("cls_id")["major_item"].apply(lambda x:list(x)[0]).reset_index()
    cls_major_item_id_list = item.rename(columns={"item":"item_list"}).groupby("cls_id")["item_list"].apply(lambda x:"|".join(list(map(str, list(set(x)))))).reset_index()


    cls_major_item_id.merge(cls_major_item_id_list, on="cls_id")

    cls_major_item_id.to_csv("cluster_major_item.csv", index=True)