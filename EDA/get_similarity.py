import argparse
import pickle
from difflib import SequenceMatcher
from time import time

import pandas as pd
from tqdm import tqdm

now = str(round(time()))[5:]


def gmp_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity(text1, text2):
    text1_set = set(text1)
    text2_set = set(text2)
    intersection = text1_set.intersection(text2_set)
    union = text1_set.union(text2_set)
    return len(intersection) / len(union)


def get_similarity_list(df: pd.DataFrame, args):
    window = args.window_size
    threshold = args.threshold
    mode = args.mode
    sim_list = []

    for i, (id, t) in tqdm(
        enumerate(
            zip(
                item.item[: len(item) - window + 1],
                item.preprocessed_title[: len(item) - window + 1],
            )
        )
    ):
        word = t
        tmp = []
        for j, (id_, t_) in enumerate(
            zip(
                item.item[i + 1 : i + window],
                item.preprocessed_title[i + 1 : i + window],
            )
        ):
            compare = t_
            if mode == 1:
                similarity = gmp_similarity(word, compare)
            elif mode == 0:
                similarity = jaccard_similarity(word, compare)
            else:
                raise Exception("mode는 0과 1 중에서 하나를 입력하세요.")
            if similarity >= threshold:
                tmp.append([word, compare, id_, similarity])
        sim_list.append(tmp)

    start = len(sim_list)
    end = len(item)

    sim_list2 = []

    # 채우지 못한 나머지 window사이즈 개 채우기
    for i, (id, t) in tqdm(
        enumerate(zip(item.item[start:end], item.preprocessed_title[start:end]))
    ):
        word = t
        tmp = []
        for j, (id_, t_) in enumerate(
            zip(
                item.item[i + start + 1 : end],
                item.preprocessed_title[i + start + 1 : end],
            )
        ):
            compare = t_
            if mode == 1:
                similarity = gmp_similarity(word, compare)
            elif mode == 0:
                similarity = jaccard_similarity(word, compare)
            else:
                raise Exception("mode는 0과 1 중에서 하나를 입력하세요.")
            if similarity >= threshold:
                tmp.append([word, compare, id_, similarity])
        sim_list2.append(tmp)

    data = sim_list[:]
    for sim in sim_list2:
        data.append(sim)

    with open(f"pkl_files/similarity_{now}.pickle", "wb") as pkl:
        pickle.dump(data, pkl)

    return data, sim_list, sim_list2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item_data_path",
        default="/opt/ml/input/data/title_preprocessed_item_93886.csv",
        type=str,
    )
    parser.add_argument(
        "--train_data_path", default="/opt/ml/input/data/hi_interaction.csv", type=str
    )
    parser.add_argument("--window_size", default=100, type=int)
    parser.add_argument("--threshold", default=0.7, type=float)
    parser.add_argument("--mode", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    item = pd.read_csv(args.item_data_path)

    item.preprocessed_title.fillna("etc", inplace=True)
    item.preprocessed_title = item.preprocessed_title.str.lower()
    item.sort_values("preprocessed_title", inplace=True)

    print(f"주의: window_size 100기준으로 약 30분 소요됩니다!")

    data, sim_list, sim_list2 = get_similarity_list(item, args)
    item["similarity_list"] = data
    item.to_csv(f"similarity_item_{now}.csv", index=False)

    print(f"★★★★ Saved similarity_item_{now}.csv ★★★★")
