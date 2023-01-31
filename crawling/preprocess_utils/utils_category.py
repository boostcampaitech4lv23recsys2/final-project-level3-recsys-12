import pandas as pd
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from konlpy.tag import Okt
from tqdm import tqdm
import re

def df_lower(df:pd.DataFrame, col:str="preprocessed_title"):
    df[col] = df[col].apply(lambda x:str.lower(x))
    return df

# TODO 전처리: [단종], [품목], (당일출고) 등등 -> "" -> 재정렬
def get_del_words(df:pd.DataFrame):
    del_words = set()
    from tqdm import tqdm
    for a in tqdm(df.preprocessed_title.unique()):
        tmp = a.strip("\t\n ")
        if tmp.startswith("[") and "]" in tmp:
            tmp = tmp.split("]")[0][1:]
            del_words.add("[" + tmp + "]")
        elif tmp.startswith("(") and ")" in tmp:
            tmp = tmp.split(")")[0][1:]
            del_words.add("(" + tmp + ")")
    return del_words

def df_strip(df:pd.DataFrame, del_words:str = "\t\n #&"):
    df.preprocessed_title = df.preprocessed_title.apply(lambda x:x.strip(del_words))
    return df

# TODO 전처리: 양끝에 \t, \n, " ", ﻿ -> ""
def df_strip2(df:pd.DataFrame):
    df.preprocessed_title = df.preprocessed_title.apply(lambda x:x.strip("\t\n ﻿"))
    return df

def df_del_word(df:pd.DataFrame, del_words:list):
    def f(x, word):
        if x.startswith(word):
            return x[len(word):].strip(" ")
        else:
            return x
    from tqdm import tqdm
    for word in tqdm(del_words):
        df.preprocessed_title = df.preprocessed_title.apply(lambda x:f(x, word))
    return df


def tokenize(x:str):
    reg = re.compile(r'[a-zA-Z]')
    okt = Okt() # 형태소 분석기 객체 생성
    noun_list = []
    x = x.split()
    for s in x:
        if reg.match(s):
            noun_list.extend(text_to_word_sequence(s))
        elif s.isdigit():
            noun_list.append(s)
        else:
            noun_list.extend(okt.nouns(okt.normalize(s)))
    return " ".join(noun_list)