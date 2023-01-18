import pandas as pd

class BaseDataLoader:
    """
    input
    path : data path
    merge_list : 추가하고자 하는 feature의 파일 명
    multi_hot : 장르를 multi_hot encoding 할 것인가
    """
    def __init__(self, path="/opt/ml/input/data/train/", merge_list=[], multi_hot=False):
        self.path = path
        self.multi_hot = multi_hot
        self.merge_list = merge_list
    
    def load(self):
        """ Load Dataset """
        self.df = pd.read_csv(self.path+"train_ratings.csv")
        for merge in self.merge_list:
            if "genres" == merge:
                continue
            self.df = pd.merge(self.df, pd.read_csv(self.path+merge+".tsv", sep="\t"))
        self.nan_processing()
        if self.multi_hot and ("genres" in self.merge_list):
            self.multi_hot_encoding()
    
    def nan_processing(self):
        """ fill na """
        if "years" not in self.merge_list:
            return
        if "titles" not in self.merge_list:
            self.df = pd.merge(self.df, pd.read_csv(self.path+"titles"+".tsv", sep="\t"))
            self.df.year = self.df.title.apply(lambda x:x[-5:-1])
            self.df.drop("title", axis=1, inplace=True)
        else:
            self.df.year = self.df.title.apply(lambda x:x[-5:-1])
    
    def multi_hot_encoding(self):
        """ genre multi-hot encoding """
        genre = pd.read_csv(self.path+"genres"+".tsv", sep="\t")
        one_hot = pd.get_dummies(genre).groupby("item").agg(lambda x:1 if sum(x)>0 else 0).reset_index().drop_duplicates()
        self.df = pd.merge(self.df, one_hot, how="left", on="item")
        