import pandas as pd
from tqdm import tqdm
import argparse
from time import time
import pickle
from utils_cluster import *
from time import time

now = str(round(time()))[5:]

CLUSTER_ID = 0
NON_CLUSTER_ID = -1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_data_path", default="similarity_item_99243.csv", type=str)
    parser.add_argument("--train_data_path", default="/opt/ml/input/data/hi_interaction.csv", type=str)
    parser.add_argument("--similarity_list_path", default="/opt/ml/input/data/similarity_99243.pickle", type=str)
    parser.add_argument("--limit_depth", default=5, type=int)
    parser.add_argument("--threshold", default=0.8, type=float)
    return parser.parse_args()

# TODO threshold 이상의 similarity 가지는 아이템을 저장하는 list 칼럼 생성
def get_item(x, threshold):
    tmp = []
    for a in x:
        if a[3] >= threshold:
            tmp.append(a[2])
    return tmp

if __name__ == "__main__":
    args = get_args()
    item = pd.read_csv(args.item_data_path)
    del_list = get_del_list()
    item.category.fillna("", inplace=True)
    with open("similarity_99243.pickle", "rb") as pkl:
        data = pickle.load(pkl)
    item["similarity_list"] = data
    item_ = item[(~item.category.str.contains("|".join(del_list))) & ~(item.category=="")]
    item_.sort_values("item", inplace=True)
    
    # TODO threshold 이상의 similarity 가지는 아이템을 저장하는 list 칼럼 생성
    def get_item(x, threshold):
        tmp = []
        for a in x:
            if a[3] >= threshold:
                tmp.append(a[2])
        return tmp

    item_["similar_item_list"] = item_.similarity_list.apply(lambda x:get_item(x, args.threshold))
    item_ = get_cluster_by_BFS(item_)
    item_.to_csv(f"clustered_item_{now}.csv", index=False)