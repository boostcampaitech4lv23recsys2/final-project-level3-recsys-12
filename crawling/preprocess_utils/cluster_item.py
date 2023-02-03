import re

import pandas as pd


class ClusterItem:
    def __init__(self, df):
        self.cluster_major_item = df

    def preprocessing(self):
        self.cluster_major_item = self.cluster_major_item.rename(
            columns={"cluster_id": "cluster_id"}
        )
        self.cluster_major_item.item_list = self.cluster_major_item.item_list.str.split(
            "|"
        )
        cluster_dict = {
            i: j
            for i, j in zip(
                self.cluster_major_item.cluster_id.values,
                self.cluster_major_item.item_list.values,
            )
        }
        tmp_list = []
        for t in cluster_dict:
            for c in cluster_dict[t]:
                tmp_list.append([t, c])
        cluster_item = pd.DataFrame(tmp_list, columns=["cluster_id", "item_id"])
        cluster_item = cluster_item.merge(
            self.cluster_major_item, on="cluster_id", how="outer"
        )
        cluster_item.drop(["item_list"], axis=1, inplace=True)

        return cluster_item


def merge_cluster(item, train, args):
    # 카테고리 필터링되면서, item에 포함되지 않은 hi_interaction item을 제거하기
    train = train[train.item.isin(item.item.unique())]

    item, item2cluster = convert_non_cluster(item)
    train["cluster_id"] = train.item.map(item2cluster)
    train.to_csv(args.output_path + "clustered_train.tsv", sep="\t", index=False)

    item = sort_by_rating_review(item)
    item.groupby("cluster_id")["item"].apply(lambda x: len(list(x)))

    cluster_major_item_id = (
        item.rename(columns={"item": "major_item"})
        .groupby("cluster_id")["major_item"]
        .apply(lambda x: list(x)[0])
        .reset_index()
    )
    cluster_major_item_id_list = (
        item.rename(columns={"item": "item_list"})
        .groupby("cluster_id")["item_list"]
        .apply(lambda x: "|".join(list(map(str, list(set(x))))))
        .reset_index()
    )

    cluster_major_item_id = cluster_major_item_id.merge(
        cluster_major_item_id_list, on="cluster_id"
    )
    cluster_major_item_id.to_csv(
        args.output_path + "cluster_major_item.tsv", sep="\t", index=False
    )
    return train, cluster_major_item_id


def convert_non_cluster(df: pd.DataFrame):
    c = df.cluster_id.max()
    cluster_ids_for_non_cluster = [
        i + c + 1 for i in range(len(df[df.cluster_id == -1]))
    ]
    df.loc[df.cluster_id == -1, "cluster_id"] = cluster_ids_for_non_cluster
    item2cluster = {
        item_id: cluster_id for item_id, cluster_id in zip(df.item, df.cluster_id)
    }
    return df, item2cluster


def sort_by_rating_review(df: pd.DataFrame):
    # 별점 기준으로 정렬하기 위해 별점 결측치를 채우는데, 대체하지는 않고 별도의 칼럼을 만듦.
    def f(x):
        pattern = re.compile(r"별점\s+(\d+\.\d+)점")
        try:
            match = pattern.search(x)
            score = float(match.group(1)) if match else None
            return score
        except:
            return 0.0

    df["rating_"] = df.rating.apply(lambda x: f(x))
    df["review_"] = df.review.fillna("리뷰 쓰기첫 리뷰 두 배 적립")

    review_list = []
    p = re.compile("[(][0-9]*[)]")
    for idx, review in enumerate(df.review_):
        m = p.findall(review)
        if len(m) == 0:
            m.append("(0)")
        review_list.append(m[0][1:-1])

    df.review_ = review_list
    df.review_ = df.review_.astype(int)
    df.rating_ = df.rating_.astype(float)

    df = df.sort_values(
        ["cluster_id", "review_", "rating_"], ascending=[True, False, False]
    )
    df.drop(columns=["review_", "rating_"], inplace=True)
    return df
