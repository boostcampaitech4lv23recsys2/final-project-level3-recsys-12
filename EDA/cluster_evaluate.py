#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# In[2]:


item = pd.read_csv("../crawling/output/item.tsv", sep="\t")


# In[3]:


item.head(1)


# In[4]:


item.category.fillna("|", inplace=True)


# In[5]:


a = []
for i in item.category.unique():
    a.extend(i.split("|"))
num_cat_multihot = len(set(a))


# In[6]:


num_cat_multihot


# In[7]:


word2id = {w:i for i, w in enumerate(sorted(list(set(a))))}
category_string2cat_id = {cstr:cid for cstr, cid in zip(item.category, item.item)}
item_id2cat_id = {iid:category_string2cat_id[c] for iid, c in zip(item.item, item.category)}


# In[8]:


word2id["LED거실등"]


# In[9]:


category_string2cat_id['가구|서랍·수납장|수납장']


# In[10]:


item_id2cat_id[1]


# In[11]:


import torch

cat_id2multihot = dict()
for c in sorted(item.category.unique()):
    dummy = torch.zeros(size=(num_cat_multihot,))
    word_ids = []
    for w in c.split("|"):
        word_ids.append(word2id[w])
    dummy[word_ids] = 1
    cat_id2multihot[category_string2cat_id[c]] = dummy.long().tolist()


# In[12]:


item.item.unique()


# In[ ]:


cat_id2multihot[item_id2cat_id[1826870]]


# In[14]:


# category multi-hot
item["category_multihot"] = item.item.apply(lambda x:cat_id2multihot[item_id2cat_id[x]])


# In[15]:


print(item.head(5)[["item", "category", "category_multihot"]])


# In[16]:


import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm(tqdm())

new_df = item["category_multihot"].apply(pd.Series)


# In[17]:


item_ = item[:]
item = item_[:]


# In[18]:


final_df = pd.concat([item, new_df], axis=1)
item = final_df.drop(["category_multihot"], axis=1)


# In[19]:


item.head(1)


# In[20]:


item.seller = item.seller.apply(lambda x: x.lower())


# In[21]:


item.head(1)


# In[22]:


item.seller.nunique()


# In[23]:


item = pd.concat([item, pd.get_dummies(item.seller)], axis=1)


# In[24]:


len(item)


# In[25]:


item.head(1)


# In[26]:


item.price.isna().sum()


# In[29]:


item.head(1)


# In[30]:


# TODO: category multi-hot과 seller one-hot으로 결정트리하기
# label은 cluster_id


# In[31]:


item.head(1)


# In[32]:


item.columns


# In[33]:


item__ = item.drop(columns=["item", "category", "rating", "review", "price", "title", "seller", "discount_rate", "image", "available_product", "predict_price", "preprocessed_title", "similarity_list"])


# In[34]:


item__.columns = item__.columns.astype(str)


# In[35]:


grp1 = item__.groupby("cluster_id").filter(lambda x: len(x) >= 3)[:-1]
grp2 = item__.groupby("cluster_id").filter(lambda x: len(x) >= 3)[1:]


# In[36]:


grp1 = grp1.groupby("cluster_id")
grp2 = grp2.groupby("cluster_id")


# In[37]:


import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# In[38]:


len(item__.columns)


# In[39]:


npitem__ = item__.drop(columns=["cluster_id"]).to_numpy()


# In[40]:


data = torch.Tensor(npitem__)


# In[41]:


from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        return sample


# In[ ]:


from tqdm import tqdm

input_dim = data.shape[1]
encoding_dim = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset(data.to(device))

batch_size = 2048
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Autoencoder(input_dim, encoding_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

pbar = tqdm(range(1000))

patience = 0
min_loss = 999999
for epoch in pbar:
    losses = []
    for i, batch in enumerate(data_loader):
        inputs = batch
        targets = batch

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    if avg_loss < min_loss:
        min_loss = avg_loss
        torch.save(model.state_dict(), "./best_model.pt")
    else:
        patience += 1
    pbar.set_postfix({'logging':f'Epoch [{epoch+1}/1000], Loss: {avg_loss}'})
    if patience > 20:
        break


# In[56]:


model = Autoencoder(input_dim, encoding_dim)
model.load_state_dict(torch.load("./_best_model.pt"))
model.eval()

rdata = model.encoder(data)
rdata = rdata.detach().numpy()


# In[57]:


label = item__.cluster_id.to_numpy()


# In[58]:


label.shape


# In[69]:


tmp = label[0]
pt = []
length = 0
for i, l in enumerate(label[1:]):
    if l == tmp:
        length += 1
    if l != tmp or i == len(label[1:]) - 1:
        pt.append((i + 1, length))
        length = 0
        tmp = l


# In[72]:


groups = []

for a in pt:
    i, length = a
    groups.append((rdata[i - 1 - length:i], label[i - 1 - length:i]))


# In[136]:


grps = []
for g in groups:
    if len(g[1]) >= 2:
        grps.append(g)


# In[137]:


len(grps)


# In[173]:


m = []
for g in grps:
    m.append(len(g[1]))


# In[174]:


sum(m) / len(m)


# In[200]:


grps.sort(key=lambda x:len(x[1]))


# In[201]:


grp1 = grps[:-1]
grp2 = grps[1:]


# In[241]:


# TODO: DecisionTreeRegressor 사용
from tqdm import tqdm

tmp = list()

for (X1, y1), (X2, y2) in tqdm(zip(grp1, grp2)):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)
    
    y1_train = [1.0] * len(y1_train)
    y1_test = [1.0] * len(y1_test)
    y2_train = [0.0] * len(y2_train)
    y2_test = [0.0] * len(y2_test)
    y1_train.extend(y2_train)
    
    X_train = np.concatenate((X1_train, X2_train), axis=0)
    # X_test = np.concatenate((X1_test, X2_test), axis=0)
    X_test = X1_test
    y_train = y1_train[:]
    y_test = y1_test[:]
    
    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)

    for p in y_pred:    
        if p < 0.5:
            # print(y_pred)
            tmp.append((y1[0]))
        break
print(len(tmp))


# In[242]:


grps3 = []
for g in groups:
    if len(g[1]) >= 5:
        grps3.append(g)


# In[243]:


grps3[0][1]


# In[205]:


item[item.cluster_id == 1039].iloc[5].image


# In[244]:


checking = []
for cluster_id in tmp:
    checking.append(item[item.cluster_id == cluster_id])


# In[257]:


for i in range(len(checking[75])):
    print(checking[75].iloc[i].image)


# In[256]:


checking[75]


# In[ ]:




