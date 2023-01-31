import pandas as pd

class House:
    def __init__(self, house):
        self.house = house
    
    def preprocessing(self):
        self.house = self.house.rename(columns={
            "공간":"space", "평수":"size", "작업":"work", 
            "분야":"category", "가족형태":"family", 
            "지역":"region", "스타일":"style", "기간":"duration",
            "예산":"budget","세부공사":"detail","좋아요":"prefer",
            "스크랩":"scrab","댓글":"comment","조회":"views"
            })

        # string 숫자 -> int(쉼표 제거)
        self.house.views = self.house.views.apply(lambda x: x.replace(',', ''))
        self.house.scrab = self.house.scrab.apply(lambda x: x.replace(',', ''))
        self.house.prefer = self.house.prefer.apply(lambda x: x.replace(',', ''))

        self.house.views = pd.to_numeric(self.house.views)
        self.house.scrab = pd.to_numeric(self.house.scrab)
        self.house.prefer = pd.to_numeric(self.house.prefer)
        
        self.house["size"] = self.house["size"].fillna("0평").apply(lambda x:int(x.rstrip("평대이상하미만 ")))
        
        return self.house