import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="data", help="data폴더 경로를 입력하세요")
    parser.add_argument("--n_iter", type=int, default=5, help="스크롤 횟수를 지정합니다.")

    parser.add_argument(
        "--target",
        nargs="+",
        type=str,
        default=["housecode", "hi", "hc", "house", "item", "card", "color"],
        help='"housecode","hi", "hc","house","item","card","color" 중 탐색을 진행 할 대상을 지정해 주세요.',
    )
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        help='"even","odd", "all" 중 탐색을 진행 할 대상을 지정해 주세요.',
    )

    parser.add_argument(
        "--just_update",
        action="store_true",
        help="이 옵션을 추가하면 추가 데이터 수집만을 진행합니다(Update만 진행합니다.) 그렇지 않을 경우 모든 데이터를 탐색합니다.",
    )
    args = parser.parse_args()

    return args


"""
전체 다 탐색 : python main.py
일부 (새로운 데이터)만 탐색 : python main.py --n_iter 10 --just_update
"""
# nohup python3 main.py --just_update —-target item > item_update.out&nohup python3 main.py --just_update —-target house > house_update.out&nohup python3 main.py --just_update —-target card > card_update.out
