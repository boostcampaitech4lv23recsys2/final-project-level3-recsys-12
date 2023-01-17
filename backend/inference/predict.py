class Model:
    def __init__(self) -> None:
        pass

    def predict(self, data):
        return self.forward(data)

    # item_id, 가구명, 가구파는 곳, 가격, 이미지 url
    def forward(self, data):
        data_dict = {'item_id' : {1,2,3,4,5,6},
        'furniture_name' : {'a','b','c','d','e'},
        'seller' : {'IKEA','LG','SamSung','Ohou','Hansam'},
        'price' : {100,100,100,100,100,100},
        "image_url" : {'a/','b/','c/','d/','e/'},}
        
        return data_dict

MODEL = Model() # TODO: model load 하는 코드로 변경

def inference(data):
    return MODEL.predict(data)
