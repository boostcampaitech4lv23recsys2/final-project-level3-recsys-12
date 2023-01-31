import pandas as pd

class ClusterItem():
    def __init__(self, df):
        self.cluster_major_item = df
    
    def preprocessing(self):
        self.cluster_major_item = self.cluster_major_item.rename(columns={"cls_id":"cluster_id"})
        self.cluster_major_item.item_list = self.cluster_major_item.item_list.str.split('|')
        cluster_dict = {i:j for i, j in zip(self.cluster_major_item.cluster_id.values, self.cluster_major_item.item_list.values)}
        tmp_list = []
        for t in cluster_dict:
            for c in cluster_dict[t]:
                tmp_list.append([t, c])
        cluster_item = pd.DataFrame(tmp_list, columns=['cluster_id', 'item_id'])
        cluster_item = cluster_item.merge(self.cluster_major_item, on="cluster_id", how="outer")
        cluster_item.drop(["item_list"], axis=1, inplace=True)
        
        return cluster_item