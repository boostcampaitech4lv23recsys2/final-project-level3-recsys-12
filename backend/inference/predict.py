import sys
import torch
import random

sys.path.append("../model/AE_model/")
from Multi_DAE import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, model_info, df) -> None:
        self.model_path = model_info["path"]["model_path"]
        self.p_dims = model_info["parameter"]["p_dims"]
        self.topk = model_info["inference"]["topk"]
        self.model = MultiDAE(
                p_dims=self.p_dims + [df.item.nunique()],
            ).to(device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.device = device
        self.house_encoder, self.house_decoder = generate_encoder_decoder(df, "house")
        self.item_encoder, self.item_decoder = generate_encoder_decoder(df, "item")
        self.dummy_input = torch.zeros((df.item.nunique()), dtype=torch.int64)
        

    def predict(self, data):
        return self.forward(data)

    # user_selected_data: label encoding되지 않은 item_id list
    def forward(self, user_selected_data):
        self.model.eval()
        tmp = []
        for item_id in user_selected_data:
            try:
                tmp.append(self.item_encoder[item_id])
            except:
                pass
        # tmp: tmp는 item_encoder에 key값으로 포함된 item들임. 
        # 만약 tmp가 빈 리스트라면, 일단 랜덤으로 채움. (아마 데이터 전처리하고 DB랑 item.csv 간의 차이가 없으면 거의 발생 안할 일로 보임)
        if not tmp: tmp = random.sample(list(self.item_decoder.keys()), k=50)
        # tmp: 실제로 모델이 inference할 수 있는 label encoding된 item들
        user_selected_data = torch.tensor(tmp)
        tmp = user_selected_data[:]
        # self.dummy_input: torch.zeros((df.item.nunique()), dtype=torch.int64): [0 0 0 0 0 0 0 ... 0 0 0 0]
        user_selected_data = self.dummy_input[:]
        user_selected_data[tmp] = 1
        # encoding된 tmp는 encoding된 item_id임. 이 encoding은 결국 item 전체의 index임. 이에 해당하는 index부분을 1로 바꿔줌
        # [0 0 0 0 0 0 0 ... 0 0 0 0] -> [1 0 0 1 00 0 0 ... 0 0 0 1]
        # [[1 0 0 1 00 0 0 ... 0 0 0 1],
        #  [1 0 0 1 00 0 0 ... 0 0 0 1],
        #  [1 0 0 1 00 0 0 ... 0 0 0 1],
        #  [1 0 0 1 00 0 0 ... 0 0 0 1]] -> user by item -> 1 by item
        # user_selected_data.unsqueeze(dim=0): [] -> [[]]: 2차원으로 넣어야함. 왜냐면 학습할 때 user by item이라는 2차원 행렬로 학습하기 때문.
        user_selected_data = user_selected_data.unsqueeze(dim=0)

        with torch.no_grad():
            prediction = []
            model_output = self.model(user_selected_data.type(torch.FloatTensor).to(self.device))
            recommend_res = model_output.argsort(dim=1).squeeze()
            up = recommend_res[-self.topk:].cpu().numpy().tolist()
            for item in up[::-1]:
                prediction.append(self.item_decoder[item])
        return prediction

def inference(data, MODEL):
    return MODEL.predict(data)
