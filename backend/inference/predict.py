import yaml
import sys
import torch

sys.path.append("/opt/ml/input/final/model/AE_model/")
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

    # item_id, 가구명, 가구파는 곳, 가격, 이미지 url
    def forward(self, user_selected_data):
        self.model.eval()
        
        # 원래는 아래 코드가 잘 동작해야 합니다
        # user_selected_data = torch.tensor(user_selected_data).apply_(lambda x: self.item_encoder[x])
        
        # 하지만, DB에서 저장된 item_id와 csv 파일에서의 item_id 간의 차이가 있어서인지
        # csv 파일에서의 item_id로 만들어진 self.item_encoder가, DB에서 불러온 item_id 간 차이로 key error 발생합니다
        tmp = []
        for item_id in user_selected_data:
            try:
                tmp.append(self.item_encoder[item_id])
            except:
                pass
        # 만약 tmp가 빈 리스트라면
        if not tmp: tmp = [k for k in list(self.item_decoder.keys())[:10]]
        user_selected_data = torch.tensor(tmp)
        tmp = user_selected_data[:]
        user_selected_data = self.dummy_input[:]
        user_selected_data[tmp] = 1
        user_selected_data = user_selected_data.unsqueeze(dim=0)

        # print(user_selected_data)
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
