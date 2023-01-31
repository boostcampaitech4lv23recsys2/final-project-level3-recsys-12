import pandas as pd
import os
from preprocess_utils.cluster_item import ClusterItem
from preprocess_utils.house import House
from preprocess_utils.item import Item
from preprocess_utils.card import Card

class Main:
    def __init__(self):
        dir_path = os.path.join("data/")
        self.house = pd.read_csv(dir_path+"house.csv")
        self.item = pd.read_csv(dir_path+"item.csv")
        self.house_item = pd.read_csv(dir_path+"hi_interaction.csv")
        self.house_card = pd.read_csv(dir_path+"hc_interaction.csv")
        self.card = pd.read_csv(dir_path+"card.csv")
        self.cluster_major_item = pd.read_csv(dir_path+"cluster_major_item.csv")
        self.clusterd_item = pd.read_csv(dir_path+"clustered_item.csv")
        
        self.preprocessed_house = None
        self.preprocessed_item = None
        self.preprocessed_clusterd_item = None
        self.preprocessed_card = None
    
    def preprocessing(self):
        _cluster_item = ClusterItem(self.cluster_major_item)
        _house = House(self.house)
        _item = Item(self.item)
        _card = Card(self.card, self.house_card)
        
        self.preprocessed_clusterd_item = _cluster_item.preprocessing()
        self.preprocessed_house = _house.preprocessing()
        self.preprocessed_item = _item.preprocessing()
        self.preprocessed_card = _card.preprocessing()

    def make_csv(self):
        os.makedirs('output', exist_ok=True)
        output_path = os.path.join('output/')
        # cluster_item은 db에 저장할 때 idx가 필요하므로 index=True로 설정
        self.preprocessed_clusterd_item.to_csv(output_path+"cluster_item.csv")
        self.preprocessed_house.to_csv(output_path+"house.tsv", sep="\t", index=False)
        self.preprocessed_item.to_csv(output_path+"item.tsv", sep="\t", index=False)
        self.preprocessed_card.to_csv(output_path+"card.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main = Main()
    main.preprocessing()
    main.make_csv()