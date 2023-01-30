import pandas as pd
import argparse
from utils_title import *
from time import time

now = str(round(time()))[5:]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_data_path", default="/opt/ml/input/data/item.csv", type=str)
    parser.add_argument("--train_data_path", default="/opt/ml/input/data/hi_interaction.csv", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    item = pd.read_csv(args.item_data_path)
    
    ##### 전처리 #####
    item["original_title"] = item.title
    item.rename(columns={"title":"preprocessed_title"}, inplace=True)
    item = df_lower(item)
    del_list = get_del_words(item)
    item = df_del_word(item, del_list)
    item = df_lower(item)
    item = df_strip(item)
    item = df_strip2(item)
    item.preprocessed_title = item.preprocessed_title.str.replace(pat=r'[^\w]',repl=r' ',regex=True)

    item.preprocessed_title.fillna("ETC", inplace=True)
    item.preprocessed_title = item.preprocessed_title.progress_apply(lambda x:tokenize(x))
    
    item.to_csv(f"title_preprocessed_item_{now}.csv", index=False)