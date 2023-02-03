import os
import pickle
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm


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
    threshold = args.similarity_threshold
    mode = args.mode
    sim_list = []

    for i, (id, t) in tqdm(
        enumerate(
            zip(
                df.item[: len(df) - window + 1],
                df.preprocessed_title[: len(df) - window + 1],
            )
        )
    ):
        word = t
        tmp = []
        for j, (id_, t_) in enumerate(
            zip(df.item[i + 1 : i + window], df.preprocessed_title[i + 1 : i + window])
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
    end = len(df)

    sim_list2 = []

    # 채우지 못한 나머지 window사이즈 개 채우기
    for i, (id, t) in tqdm(
        enumerate(zip(df.item[start:end], df.preprocessed_title[start:end]))
    ):
        word = t
        tmp = []
        for j, (id_, t_) in enumerate(
            zip(
                df.item[i + start + 1 : end], df.preprocessed_title[i + start + 1 : end]
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

    os.makedirs("pkl", exist_ok=True)
    if args.test:
        with open(args.output_path + "similarity.pickle", "wb") as pkl:
            pickle.dump(data, pkl)
    else:
        with open(args.similarity_list_path, "wb") as pkl:
            pickle.dump(data, pkl)

    return data, sim_list, sim_list2
