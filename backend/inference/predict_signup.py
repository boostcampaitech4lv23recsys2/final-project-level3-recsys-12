
# DB
from db.models import Card
from db.db_connect import Database
from sqlalchemy import select
from sqlalchemy import func

# DL Model
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Image utils
import requests
from io import BytesIO
from PIL import Image, ImageFile
# import matplotlib.pyplot as plt

'''
Input : [유저가 선택한 카드번호 리스트, 유저가 입력한 회원정보]
Output : 유저가 선택한 카드id 리스트 (10개)
'''
database = Database()

class Config:
    def __init__(self):
        self.epoch = 30
        self.csv_path = "./data"
        self.img_path = "./card"
        self.model_path="./inference/card_vectorizer.pth"
        self.card_vector_path="./inference/card_vector.csv"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.num_workers = 6
        self.lr = 0.001

        # AE
        self.width = 224
        self.height = 224
        self.color_channel = 3
        self.hidden_dim = 2048
        self.latent_dim = 16
        self.noise_mean = 0.2
        self.noise_std = 0.1
        self.dropout = 0.3

        # CAE
        self.in_channel = 3
        self.l1 = 8
        self.l2 = 16
        self.l3 = 32
        self.l4 = 64
        self.l5 = 128
        self.hidden_dim = 1024
        self.latent_dim = 64

        #util
        self.figsize = (3,3)


class ImageUtil:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    
    def load_image_by_url(self, url):
        res = requests.get(url)
        request_get_img = Image.open(BytesIO(res.content)).convert("RGB")
        return self.transform(request_get_img)
        
    def load_image_by_card_num(self, card_num):
        res = os.path.join(self.config.img_path, f"{card_num}.jpg")
        request_get_img = Image.open(res).convert("RGB")
        return self.transform(request_get_img)
    
    # def imshow(self, img, title=""):
    #     img = img / 2 + 0.5
    #     npimg = img.numpy()
    #     fig = plt.figure(figsize=self.config.figsize)
    #     plt.title(title)

    #     plt.imshow(np.transpose(npimg, (1,2,0)))
    #     plt.show()
        

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        selected = self.df.iloc[idx]
        img = self.image_util.load_image_by_card_num(selected.card)
        style = selected[self.style_categories]
        return img, selected.card

class ConvAutoEncoder(nn.Module):
    def __init__(self, config):
        super(ConvAutoEncoder,self).__init__()
        self.in_channel = config.in_channel
        self.l1 = config.l1
        self.l2 = config.l2
        self.l3 = config.l3
        self.l4 = config.l4
        self.l5 = config.l5
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim

        self.encoder = nn.Sequential(
            self.encoder_cnn(),
            nn.Flatten(start_dim=1),
            self.encoder_linear()
        )
        self.decoder = nn.Sequential(
            self.decoder_linear(),
            nn.Unflatten(dim=1, unflattened_size=(self.l5,7,7)),
            self.decoder_cnn(),
            nn.Sigmoid()
        )

    def encoder_cnn(self):
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        return nn.Sequential(
            nn.Conv2d(self.in_channel, l1, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(l1, l2, 3, stride=2, padding=1),
            nn.BatchNorm2d(l2),
            nn.ReLU(),
            nn.Conv2d(l2, l3, 3, stride=2, padding=1),
            nn.BatchNorm2d(l3),
            nn.ReLU(),
            nn.Conv2d(l3, l4, 3, stride=2, padding=1),
            nn.BatchNorm2d(l4),
            nn.ReLU(),
            nn.Conv2d(l4, l5, 3, stride=2, padding=1),
            nn.ReLU()
        )
    
    def encoder_linear(self):
        return nn.Sequential(
            nn.Linear(7*7*self.l5,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
    def decoder_linear(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 7*7*self.l5),
            nn.ReLU()
        )
    
    def decoder_cnn(self):
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        return nn.Sequential(
            nn.ConvTranspose2d(l5, l4, 2, stride=2, output_padding=0),
            nn.BatchNorm2d(l4),
            nn.ReLU(),
            nn.ConvTranspose2d(l4, l3, 2, stride=2, output_padding=0),
            nn.BatchNorm2d(l3),
            nn.ReLU(),
            nn.ConvTranspose2d(l3, l2, 2, stride=2, output_padding=0),
            nn.BatchNorm2d(l2),
            nn.ReLU(),
            nn.ConvTranspose2d(l2, l1, 2, stride=2, output_padding=0),
            nn.BatchNorm2d(l1),
            nn.ReLU(),
            nn.ConvTranspose2d(l1, self.in_channel, 2, stride=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded        

class CardVectorizer:
    def __init__(self, config):
        self.config = config
        self.model_path = config.model_path
        self.card_vector_path = config.card_vector_path
        self.util = ImageUtil(config)

    def load_model(self):
        self.model = ConvAutoEncoder(self.config)
        statedict = torch.load(self.model_path)
        self.model.load_state_dict(statedict)
        
    def load_data(self):
        self.card_vector_df = pd.read_csv(self.card_vector_path)
        self.card2vector = {}
        for card_num in tqdm(self.card_vector_df.columns):
            self.card2vector[int(card_num)] = self.card_vector_df[card_num].tolist()
        self.card2idx = {j:i for i, j in enumerate(self.card2vector)}
        self.idx2card = {j:i for i, j in self.card2idx.items()}
        self.entire_card_length = len(self.card2idx)

    def sampling_cards(self, selected_list:list=[], sampling_size=10, narrow_size=3, **kwargs):
        selected_len = len(selected_list)
        sampling_grid = max(self.entire_card_length // (narrow_size ** 2),sampling_size*2)
        card_vectors = torch.Tensor([self.card2vector[i] for i in selected_list])
        user_vector = self.vector_mean(card_vectors)
        distance = torch.Tensor([self.get_norm(user_vector,  torch.Tensor(card_vector)) for card_vector in self.card2vector.values()])
        distance_idx = torch.argsort(distance, dim=0)[:sampling_grid]
        sampled_card_idx = distance_idx[torch.randperm(distance_idx.size(0))[:sampling_size]]
        return [self.idx2card[int(i)] for i in sampled_card_idx]
    
    def vector_mean(self, vectors):
        return torch.sum(vectors, axis=0)/len(vectors)

    def get_norm(self, vector1, vector2):
        return torch.norm(vector1-vector2)

class Signup_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
    # def __init__(self, card_list, member_info) -> None:
    #     self.card_list = card_list
    #     self.member_info = member_info

    def forward(self, card_id, space, size, family):
        
        with database.session_maker() as session:
            stmt = select(Card).where(Card.img_space =='거실' and Card.is_human==0).order_by(func.random()).limit(10)
            data = session.execute(stmt).fetchall()
        return [col.Card.card_id for col in data]
    
model = Signup_Model()
def inference_signup(card_id, space, size, family):
    return model(card_id, space, size, family)