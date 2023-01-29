import os
import pandas as pd
from tqdm import tqdm
from crawling_modules import Bs4Crawler, SeleniumCrawler, TodaysHome
from args import get_args
from collections import defaultdict
from utils import logging_time, merge_all_in_dir

class CrawlingManager(TodaysHome):
    """새로 등록된 house, item, card에 대한 정보를 업데이트 합니다."""
    @logging_time
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hi_interaction_folder = os.path.join(self.args.datapath, "hi_interaction")
        self.hc_interaction_folder = os.path.join(self.args.datapath, "hc_interaction")

        self.house_code_path = os.path.join(self.args.datapath, "house_code.csv")
        self.hi_interaction_path = os.path.join(self.args.datapath, "hi_interaction.csv")
        self.hc_interaction_path = os.path.join(self.args.datapath, "hc_interaction.csv")
        self.house_path = os.path.join(self.args.datapath, "house.csv")
        self.item_path = os.path.join(self.args.datapath, "item.csv")
        
        self.main_color_path = os.path.join(self.args.datapath, "house_main_color.csv")
        self.wall_color_path = os.path.join(self.args.datapath, "house_wall_color.csv")
        self.floor_color_path = os.path.join(self.args.datapath, "house_floor_color.csv")

        self.original_house_code = pd.read_csv(self.house_code_path)
        self.original_hi_interaction = pd.read_csv(self.hi_interaction_path)
        self.original_hc_interaction = pd.read_csv(self.hc_interaction_path)
        self.original_house = pd.read_csv(self.house_path)
        self.original_item = pd.read_csv(self.item_path)
        
        self.pre_house_code = set(map(int, self.original_house.house.values))
        self.pre_item_code = set(map(int, self.original_item.item.values))
        
        self.new_house = set()
        self.new_item = set()

        self.crawler = SeleniumCrawler()
        self.bs4_crawler = Bs4Crawler()
        

        if self.args.type=="all":
            self.card_path = os.path.join(self.args.datapath, "card.csv")
        elif self.args.type=="even":
            self.card_path = os.path.join(self.args.datapath, "cardeven.csv")
        elif self.args.type=="odd":
            self.card_path = os.path.join(self.args.datapath, "cardodd.csv")
        else:
            pass
        
        self.original_card = pd.read_csv(self.card_path)
        self.pre_card_code = set(map(int, self.original_card.card.values))
        self.new_card = set()

        self.func_dict = {
            "housecode":self.update_house_code,
            "hi":self.update_hi_interaction,
            "hc":self.update_hc_interaction,
            "house":self.update_house,
            "item":self.update_item,
            "card":self.update_card,
            "color":self.get_color,
        }

    @logging_time
    def update_house_code(self):
        """house code를 탐색합니다."""
        self.crawler.get_page(self.crawler.get_project_list_url())
        if self.args.just_update:
            updated_house = self.crawler.scroll_down(class_name=self.house_in_projects_class, n_iter=self.args.n_iter)
        else:
            updated_house = self.crawler.scroll_down(class_name=self.house_in_projects_class)
        new = pd.DataFrame(set(map(int,updated_house)),columns=["house"])
        pd.concat([self.original_house_code, new]).drop_duplicates().sort_values("house").to_csv(self.house_code_path,index=False)
        self.original_house_code = pd.read_csv(self.house_code_path)
    
    @logging_time
    def update_hi_interaction(self):
        """house의 상품 모아보기를 탐색합니다."""
        entire_house_code = set(self.original_house_code.house.astype(int).values)
        setted = set(map(lambda x:int(x.split("_")[-1].split(".")[0]),os.listdir(self.hi_interaction_folder)))
        founded_house = entire_house_code - setted
        iterator = tqdm(founded_house, "hi_interaction") if self.args.just_update else tqdm(entire_house_code, "hi_interaction")
        for housecode in iterator:
            file_name = os.path.join(self.hi_interaction_folder, f"house_item_interaction_{housecode}.csv")
            updated_data = []
            self.crawler.get_page(self.crawler.get_project_use_products_url(housecode))
            items = map(int, self.crawler.scroll_down(class_name=self.item_in_user_products_class))
            for item in items:
                updated_data.append([housecode, item])
            new_data = pd.DataFrame(updated_data, columns=["house","item"])
            if housecode in setted:
                origin = pd.read_csv(file_name)
                pd.concat([origin, new_data]).drop_duplicates().sort_values("item").to_csv(file_name, index=False)
            else:
                new_data.drop_duplicates().sort_values("item").to_csv(file_name, index=False)                
        merge_all_in_dir(self.hi_interaction_folder).sort_values(["house","item"]).to_csv(self.hi_interaction_path, index=False)
        self.original_hi_interaction = pd.read_csv(self.hi_interaction_path)

    @logging_time
    def update_hc_interaction(self):
        """house의 card(집 사진)을 탐색합니다."""
        entire_house_code = set(self.original_house_code.house.astype(int).values)
        setted = set(map(lambda x:int(x.split("_")[-1].split(".")[0]),os.listdir(self.hc_interaction_folder)))
        founded_house = entire_house_code - setted
        iterator = tqdm(founded_house, "hc_interaction") if self.args.just_update else tqdm(entire_house_code, "hc_interaction")
        for housecode in iterator:
            if housecode in setted:
                continue
            file_name = os.path.join(self.hc_interaction_folder, f"house_card_interaction_{housecode}.csv")
            updated_data = []
            self.bs4_crawler.get_page(self.bs4_crawler.get_project_url(housecode))
            if self.bs4_crawler.page:
                card_elements = self.bs4_crawler.page.find_all("a","project-detail-image-block__link")
                for card_data in card_elements:
                    card = int(card_data.attrs["href"].split("/")[-1])
                    updated_data.append([housecode, card])
            pd.DataFrame(updated_data, columns=["house","card"]).to_csv(file_name, index=False)
        merge_all_in_dir(self.hc_interaction_folder).sort_values(["house","card"]).to_csv(self.hc_interaction_path, index=False)
        self.original_hc_interaction = pd.read_csv(self.hc_interaction_path)

    @logging_time
    def update_house(self):
        """house의 정보를 탐색합니다."""
        entire_house_code = set(self.original_house_code.house.astype(int).values)
        setted = set(self.original_house.house.astype(int).values)
        founded_house = entire_house_code - setted
        new = {}
        iterator = tqdm(founded_house,"house") if self.args.just_update else tqdm(entire_house_code,"house")
        for house in iterator:
            self.bs4_crawler.get_page(self.bs4_crawler.get_project_url(house))
            data = self.bs4_crawler.get_features()
            if data:
                new[house] = data
            new_df = pd.DataFrame.from_dict(new, orient="index").reset_index().rename(columns={"index":"house"})
            pd.concat([self.original_house, new_df]).drop_duplicates().sort_values("house").to_csv(self.house_path,index=False)

    @logging_time
    def update_item(self):
        """item 정보를 탐색합니다."""
        entire_item_code = set(self.original_hi_interaction.item.astype(int).values)
        setted = set(self.original_item.item.astype(int).values)
        founded_item = entire_item_code - setted
        new = {}
        iterator = tqdm(founded_item,"item") if self.args.just_update else tqdm(entire_item_code,"item")
        for item in iterator:
            self.bs4_crawler.get_page(self.bs4_crawler.get_item_url(item))
            data = self.bs4_crawler.get_features()
            if data:
                new[item] = data
            new_df = pd.DataFrame.from_dict(new, orient="index").reset_index().rename(columns={"index":"item"})
            pd.concat([self.original_item, new_df]).drop_duplicates().sort_values("item").to_csv(self.item_path,index=False)
    
    @logging_time
    def update_card(self):
        """card 정보를 탐색합니다."""
        entire_card_code = set(self.original_hc_interaction.card.astype(int).values)
        setted = set(self.original_card.card.astype(int).values)
        founded_item = entire_card_code - setted
        new = {}
        if not self.args.just_update:
            tqdm(entire_card_code,"card")
        elif self.args.type == "even":
            iterator = tqdm(set(filter(lambda x:x%2==0, founded_item)),"card")
        elif self.args.type == "odd":
            iterator = tqdm(set(filter(lambda x:x%2==1, founded_item)),"card")
        elif self.args.type == "all":
            iterator = tqdm(founded_item, "card")
        else:
            pass
        # iterator = tqdm(founded_item,"card") if self.args.just_update else tqdm(entire_card_code,"card")
        for card in iterator:
            self.bs4_crawler.get_page(self.bs4_crawler.get_card_url(card))
            data = self.bs4_crawler.get_features()
            if data:
                new[card] = data
            new_df = pd.DataFrame.from_dict(new, orient="index").reset_index().rename(columns={"index":"card"})
            pd.concat([self.original_card, new_df]).drop_duplicates().sort_values("card").to_csv(self.card_path,index=False)

    @logging_time
    def get_color(self, type="main"):
        color_dict = defaultdict(lambda : {i:0 for i in range(13)})
        for i in range(13):
            self.crawler.get_page(self.get_color_url(type=type, color_num=i))
            founded = self.crawler.scroll_down(class_name="project-feed__item__link")
            for house_code in set(map(int, founded)):
                color_dict[house_code][i] = 1
        return color_dict

@logging_time
def save_color(manager):
    columns = ["house"]+[str(i) for i in range(13)]
    pd.DataFrame.from_dict(manager.get_color("main"), orient="index").reset_index().rename(columns={"index":"house"}).to_csv(manager.main_color_path,index=False)
    pd.DataFrame.from_dict(manager.get_color("wall"), orient="index").reset_index().rename(columns={"index":"house"}).to_csv(manager.wall_color_path,index=False)
    pd.DataFrame.from_dict(manager.get_color("floor"), orient="index").reset_index().rename(columns={"index":"house"}).to_csv(manager.floor_color_path,index=False)

@logging_time
def process(manager):
    for func_name in manager.args.target:
        manager.func_dict[func_name]()


if __name__ == "__main__":
    args = get_args()
    print(args)
    manager = CrawlingManager(args)
    process(manager)