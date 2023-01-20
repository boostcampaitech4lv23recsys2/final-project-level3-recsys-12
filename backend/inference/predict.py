class Model:
    def __init__(self) -> None:
        pass

    def predict(self, data):
        return self.forward(data)

    # item_id, 가구명, 가구파는 곳, 가격, 이미지 url
    def forward(self, data):
        '''
        data: 유저가 선택한 선호하는 집들이의 가구들
        '''
        # data_dict = {'item' : {1,4,5,6,8,9},
        # 'furniture_name' : {'a','b','c','d','e'},
        # 'seller' : {'IKEA','LG','SamSung','Ohou','Hansam'},
        # 'price' : {100,100,100,100,100,100},
        # "image_url" : {'a/','b/','c/','d/','e/'},}
        item_list = [1, 4, 5, 6, 8, 9, 10, 15, 16, 18, 19]
        
        return item_list

MODEL = Model() # TODO: model load 하는 코드로 변경

def inference(data):
    return MODEL.predict(data)
