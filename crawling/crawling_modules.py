import time
from urllib.request import urlopen

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

from config import Features

from utils import logging_time

class TodaysHome():
    "오늘의 집 사이트 관련 정보를 관리하는 클래스 입니다."
    def __init__(self):
        self.main_page_url = "https://ohou.se"
        self.house_in_projects_class = "project-feed__item__link"
        self.item_in_user_products_class = "production-item__overlay"
        self.item_in_user_products_wrapper_class = "e1g4zwbe0"
        self.current_project_num = None
        self.current_item_num = None
        
    def get_project_list_url(self):
        "전체 Project 리스트를 볼 수 있는 페이지 링크를 반환합니다."
        return self.main_page_url + "/projects"

    def get_project_list_by_color(self, color_idx, color_type):
        "전체 Project를 (전체, 벽, 바닥)의 색상으로 필터링 한 페이지 링크를 반환합니다."
        if color_idx < 0 or color_idx > 12:
            raise "color_idx는 0~12의 정수입니다."
        if color_type not in ("main", "floor", "wall"):
            raise "color_type은 main, floor, wall 중 하나입니다."
        return f"{self.get_project_list_url}?{color_type}={color_idx}"        

    def get_project_url(self, project_num):
        "특정 Project의 링크를 반환합니다."
        self.current_project_num = project_num
        return f"{self.get_project_list_url()}/{project_num}"
    
    def get_item_url(self, item_num):
        "특정 Item의 링크를 반환합니다."
        self.current_item_num = item_num
        return f"{self.main_page_url}/productions/{item_num}/selling"

    def get_project_use_products_url(self, project_num):
        "특정 Project에서 사용된 item들이 정리되어 있는 (상품 모아보기) 링크를 반환합니다."
        self.current_project_num = project_num
        return f"{self.get_project_list_url()}/{project_num}/use_products"

    def get_card_url(self, card_num):
        self.current_card_num = card_num
        return f"{self.main_page_url}/cards/{card_num}"

    def get_color_url(self, color_num:int, type:str):
        return f"{self.get_project_list_url()}?{type}={color_num}"

class SeleniumCrawler(TodaysHome):
    "Selenium을 사용한 크롤링 모듈입니다. house code를 모두 뽑아온다거나 house-item interaction을 모두 뽑아오는 용도로 사용할 수 있습니다."
    def __init__(self):
        super().__init__()
        self.set_options()
        self.driver = self.get_driver()

    def set_options(self):
        "Selenium에서 사용할 chrome_driver에 대한 설정을 진행합니다."
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--start-fullscreen')
        self.chrome_options.add_argument('--blink-settings=imagesEnabled=false')
    
    def get_driver(self):
        "Selenium에서 사용할 Chrome_driver를 불러옵니다."
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
        driver.implicitly_wait(time_to_wait=1)
        return driver
    
    def get_page(self, url):
        "driver로 입력받은 주소에 접근합니다."
        self.driver.get(url)

    @logging_time
    def scroll_down(self, class_name, n_iter=2**32-1)->set:
        "스크롤을 내리며 모든 상품의 href를 찾고 반홥합니다."
        delay_cnt = 0
        items = set(list(map(lambda x:x.get_attribute("href").split("/")[4], self.driver.find_elements(By.CLASS_NAME,class_name))))
        prev_height = self.driver.execute_script("return document. body.scrollHeight")

        for _ in range(n_iter):
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
            time.sleep(0.1)
            current_height = self.driver.execute_script("return document. body.scrollHeight")
            if current_height == prev_height:
                if delay_cnt > 3:
                    break
                delay_cnt += 1
                continue
            else:
                delay_cnt = 0
            prev_height = current_height
            time.sleep(0.3)
            try:
                new_items = set(list(map(lambda x:x.get_attribute("href").split("/")[4], self.driver.find_elements(By.CLASS_NAME,class_name))))
            except:
                time.sleep(0.5)
                new_items = set(list(map(lambda x:x.get_attribute("href").split("/")[4], self.driver.find_elements(By.CLASS_NAME,class_name))))
            items = items.union(new_items)
            print(f"{class_name}>>\t\tfound {len(items)} elements!")
        return items


class Bs4Crawler(TodaysHome):
    "정적 웹페이지 크롤링에 사용할 수 있습니다. house나 item의 detail information을 추출할 때 사용할 수 있습니다."
    def __init__(self):
        super().__init__()
        self.feature = Features().feature_dict

    def get_page(self, url):
        try:
            html = urlopen(url)
            # time.sleep(0.5)
            self.page = BeautifulSoup(html, "html.parser")
        except:
            self.page = None

    def get_features(self):
        feature = Features().feature_dict
        items = {}
        for col in feature:
            try:
                if feature[col].extract_method == "text":
                    items[col] = self.page.find(feature[col].sementic_tag, feature[col].class_name).text
                elif feature[col].extract_method == "special":
                    if col in ("detail_table", "review_table"):
                        table = self.page.find_all(feature[col].sementic_tag, feature[col].class_name)
                        for field in table:
                            items[field.find("dt").text] = field.find("dd").text
                    elif col == "category":
                        category_field = self.page.find(feature[col].sementic_tag, feature[col].class_name)
                        items[col] = '|'.join(list(map(lambda x:x.text, category_field.find_all("li"))))
                elif feature[col].extract_method.startswith("attrs"):
                    items[col] = self.page.find(feature[col].sementic_tag, feature[col].class_name).attrs[feature[col].extract_method.split('.')[-1]]
            except:
                continue
        return items
        