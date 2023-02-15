#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


item = pd.read_csv("../crawling/output/item.tsv", sep="\t")
train = pd.read_csv("../backend/data/train.tsv", sep="\t")
item.drop(index=item[item.rating == 'ņĀÉ 5.0ņĀ'].index, inplace=True)
item.drop(index=item[item.rating == 'м†Р 4.8м†'].index, inplace=True)
item.review = item.review.astype(int)
item.rating = item.rating.astype(float)


# In[8]:


# import webbrowser
# import time

# for a in (item[item.item.isin(top)].image):
#     webbrowser.open(a)


# In[12]:


train.item.value_counts()


# In[17]:


import matplotlib.pyplot as plt

item_counts = train['item'].value_counts()
plt.hist(item_counts, bins=50, log=True, orientation='horizontal')
plt.ylabel('Number of Interactions')
plt.xlabel('Number of Items')
plt.title('Interactions per Item (Log Scale)')
plt.show()

item_counts = train['item'].value_counts()
plt.hist(item_counts, bins=50, log=False, orientation='horizontal')
plt.ylabel('Number of Interactions')
plt.xlabel('Number of Items')
plt.title('Interactions per Item (Original Scale)')
plt.show()


# In[18]:


train_moviedata = pd.read_csv("../crawling/output/train_moviedata.csv")


# In[19]:


import matplotlib.pyplot as plt

item_counts = train_moviedata['item'].value_counts()
plt.hist(item_counts, bins=50, log=True, orientation='horizontal')
plt.ylabel('Number of Interactions')
plt.xlabel('Number of Items')
plt.title('Interactions per Item (Log Scale)')
plt.show()

item_counts = train_moviedata['item'].value_counts()
plt.hist(item_counts, bins=50, log=False, orientation='horizontal')
plt.ylabel('Number of Interactions')
plt.xlabel('Number of Items')
plt.title('Interactions per Item (Log Scale)')
plt.show()


# In[20]:


item.drop(index=item[item.rating == 'ņĀÉ 5.0ņĀ'].index, inplace=True)
item.drop(index=item[item.rating == 'м†Р 4.8м†'].index, inplace=True)
item.review = item.review.astype(int)
item.rating = item.rating.astype(float)


# In[21]:


from sklearn.preprocessing import *

mm = MinMaxScaler()
item_review = item.review.to_frame()
item["mm_review"] = mm.fit_transform(item_review).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("MinMax Scaler Transformation")
plt.show()


# In[22]:


type(mm.fit_transform(item_review))


# In[23]:


mm.fit_transform(item_review).reshape(-1)


# In[24]:


item = item[(item.rating > 0) & (item.review > 200)]


# In[25]:


from sklearn.preprocessing import *

mm = MinMaxScaler()
item_review = item.review.to_frame()
item["mm_review"] = mm.fit_transform(item_review).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("MinMax Scaler Transformation")
plt.show()


# In[26]:


from sklearn.preprocessing import *

mm = RobustScaler()
item_review = item.review.to_frame()
item["mm_review"] = mm.fit_transform(item_review).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("Robust Scaler Transformation")
plt.show()


# In[27]:


from sklearn.preprocessing import *

mm = StandardScaler()
item_review = item.review.to_frame()
item["mm_review"] = mm.fit_transform(item_review).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("Standard Scaler Transformation")
plt.show()


# In[28]:


from sklearn.preprocessing import *

mm = Normalizer()
item_review = item.review.to_frame()
item["mm_review"] = mm.fit_transform(item_review).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_review"], bins=50, edgecolor='black')
plt.xlabel("Scaled Review")
plt.ylabel("Frequency")
plt.title("Normalizer Scaler Transformation")
plt.show()


# In[29]:


from sklearn.preprocessing import *

mm = StandardScaler()
item_rating = item.rating.to_frame()
item["mm_rating"] = mm.fit_transform(item_rating).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["rating"], bins=50, edgecolor='black')
plt.xlabel("Scaled Rating")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_rating"], bins=50, edgecolor='black')
plt.xlabel("Scaled Rating")
plt.ylabel("Frequency")
plt.title("Standard Scaler Transformation")
plt.show()


# In[30]:


from sklearn.preprocessing import *

mm = MinMaxScaler()
item_rating = item.rating.to_frame()
item["mm_rating"] = mm.fit_transform(item_rating).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["rating"], bins=50, edgecolor='black')
plt.xlabel("Scaled Rating")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_rating"], bins=50, edgecolor='black')
plt.xlabel("Scaled Rating")
plt.ylabel("Frequency")
plt.title("MinMax Scaler Transformation")
plt.show()


# In[31]:


from sklearn.preprocessing import *

mm = RobustScaler()
item_rating = item.rating.to_frame()
item["mm_rating"] = mm.fit_transform(item_rating).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["rating"], bins=50, edgecolor='black')
plt.xlabel("Scaled Rating")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_rating"], bins=50, edgecolor='black')
plt.xlabel("Scaled Rating")
plt.ylabel("Frequency")
plt.title("Robust Scaler Transformation")
plt.show()


# In[32]:


from sklearn.preprocessing import *

mm = Normalizer()
item_rating = item.rating.to_frame()
item["mm_rating"] = mm.fit_transform(item_rating).squeeze(1)

import matplotlib.pyplot as plt

plt.hist(item["rating"], bins=50, edgecolor='black')
plt.xlabel("Scaled Rating")
plt.ylabel("Frequency")
plt.title("NO Scaler Transformation")
plt.show()

plt.hist(item["mm_rating"], bins=50, edgecolor='black')
plt.xlabel("Scaled Rating")
plt.ylabel("Frequency")
plt.title("Noramlizer Scaler Transformation")
plt.show()


# In[33]:


import torch
torch.as_tensor(item.rating.to_numpy()).float()


# In[34]:


train


# In[35]:


t = [33050, 441917, 212240, 876149, 30484, 41712, 853362, 30495, 33081, 30504, 33086, 1152, 38925, 33089, 30528, 639470, 38945, 30532, 38947, 38951, 563197, 38953, 26926, 38963, 909706, 909717, 315076, 544635, 1320247, 36594, 48152, 768611, 386975, 176428, 14395, 61315, 1092386, 225061, 387093, 387106, 33580, 748742, 123463, 310209, 1321586, 33735, 280948, 135891, 1055509, 303875, 18446, 131781, 1325, 1036195, 131838, 31085, 33852, 68422, 680807, 481893, 537573, 225482, 213839, 225484, 310706, 172680, 42545, 136673, 1450530, 641627, 181848, 88674, 34103, 414026, 1451760, 31336, 128435, 691069, 735254, 414449, 24302, 109802, 348334, 1026007, 1399771, 69056, 1154849, 148603, 2699, 354736, 182232, 226085, 567071, 89085, 34624, 547677, 624613, 155409, 451187, 1071454, 246017, 11721, 11727, 97064, 11729, 451232, 37307, 11733, 306151, 11738, 11731, 429064, 11750, 823099, 173544, 363774, 701233, 1072212, 1072221, 752541, 2864, 1006410, 19486, 79841, 1006485, 438844, 237045, 149087, 1171337, 237059, 1157216, 1060151, 485244, 1248819, 19562, 1028902, 773248, 1060535, 938129, 65810, 814245, 31965, 251331, 981458, 1084568, 58467, 312920, 15735, 872870, 702583, 15833, 169066, 605321, 49639, 12589, 1158159, 585078, 237554, 313120, 753989, 682, 32125, 313203, 169244, 153526, 705, 16201, 540903, 745550, 138894, 204610, 71092, 29275, 204619, 995584, 995602, 995590, 204611, 313488, 336837, 204615, 32199, 13082, 158184, 46969, 826577, 1235652, 328613, 307885, 1020795, 252272, 874418, 264565, 1160354, 170039, 36011, 550989, 675895, 884208, 83763, 130321, 308338, 117522, 38483, 13236, 954778, 1076749, 153835, 766950, 224146, 1414013, 13495, 13505, 13524, 731281, 747576, 17023, 38044, 30335, 1276387, 1276391]


# In[36]:


item[(item.item.isin(t)) & (item.review > 0)]


# In[37]:


item[item.item.isin([548202, 46969, 701233, 441917, 212240, 33050, 481893, 1072221, 230026, 33081, 1020795, 59672, 225482, 438844, 641627, 382544, 514543, 884208, 65810, 639876, 196558, 735254, 154085, 289321, 230603, 176428, 414504, 313120, 753989, 354736, 543120, 313203, 543130, 543131, 543132, 547677, 543134, 123463, 138894, 204610, 365163, 71092, 204619, 451187, 543133, 995602, 210373, 11721, 97064, 479517])]


# In[38]:


item[item.item.isin([204615, 745550, 550989, 481893, 246017, 176428, 97064, 582913, 237059, 225484, 313120, 747576, 753989, 544635, 204610])]


# In[39]:


# import webbrowser
# import time

# # t = [204615, 745550, 550989, 481893, 246017, 176428, 97064, 582913, 237059, 225484, 313120, 747576, 753989, 544635, 204610]
# t = [33050, 441917, 212240, 876149, 30484, 41712, 853362, 30495, 33081, 30504, 33086, 1152, 38925, 33089, 30528, 639470, 38945, 30532, 38947, 38951, 563197, 38953, 26926, 38963, 909706, 909717, 315076, 544635, 1320247, 36594, 48152, 768611, 386975, 176428, 14395, 61315, 1092386, 225061, 387093, 387106, 33580, 748742, 123463, 310209, 1321586, 33735, 280948, 135891, 1055509, 303875, 18446, 131781, 1325, 1036195, 131838, 31085, 33852, 68422, 680807, 481893, 537573, 225482, 213839, 225484, 310706, 172680, 42545, 136673, 1450530, 641627, 181848, 88674, 34103, 414026, 1451760, 31336, 128435, 691069, 735254, 414449, 24302, 109802, 348334, 1026007, 1399771, 69056, 1154849, 148603, 2699, 354736, 182232, 226085, 567071, 89085, 34624, 547677, 624613, 155409, 451187, 1071454, 246017, 11721, 11727, 97064, 11729, 451232, 37307, 11733, 306151, 11738, 11731, 429064, 11750, 823099, 173544, 363774, 701233, 1072212, 1072221, 752541, 2864, 1006410, 19486, 79841, 1006485, 438844, 237045, 149087, 1171337, 237059, 1157216, 1060151, 485244, 1248819, 19562, 1028902, 773248, 1060535, 938129, 65810, 814245, 31965, 251331, 981458, 1084568, 58467, 312920, 15735, 872870, 702583, 15833, 169066, 605321, 49639, 12589, 1158159, 585078, 237554, 313120, 753989, 682, 32125, 313203, 169244, 153526, 705, 16201, 540903, 745550, 138894, 204610, 71092, 29275, 204619, 995584, 995602, 995590, 204611, 313488, 336837, 204615, 32199, 13082, 158184, 46969, 826577, 1235652, 328613, 307885, 1020795, 252272, 874418, 264565, 1160354, 170039, 36011, 550989, 675895, 884208, 83763, 130321, 308338, 117522, 38483, 13236, 954778, 1076749, 153835, 766950, 224146, 1414013, 13495, 13505, 13524, 731281, 747576, 17023, 38044, 30335, 1276387, 1276391]

# for a in (item[(item.item.isin(t)) & (item.review > 0)].image):
#     webbrowser.open(a)
# # item[(item.item.isin(t)) & (item.review > 0)]


# In[40]:


item[(item.item.isin(t)) & (item.review > 0)]


# In[41]:


import os

data_path = "../backend/data/"

train = (
    pd.read_csv(os.path.join(data_path, "train.tsv"), sep="\t")
    .groupby("house")
    .filter(lambda x: len(x) >= 15)
)
item = (
    pd.read_csv(os.path.join(data_path, "item.tsv"), sep="\t")
)
a = item[item.rating == 'ņĀÉ 5.0ņĀ']
b = item[item.rating == 'м†Р 4.8м†']
tmp = list(a.index)
tmp.extend(list(b.index))

tmp2 = list(train[train.item == a.item.values[0]].index)
tmp2.extend(list(train[train.item == b.item.values[0]].index))

item.drop(index=tmp, inplace=True)
train.drop(index=tmp2, inplace=True)


# In[42]:


train.item.nunique()


# In[43]:


import os

data_path = "../backend/data/"

train = (
    pd.read_csv(os.path.join(data_path, "train.tsv"), sep="\t")
)
item = (
    pd.read_csv(os.path.join(data_path, "item.tsv"), sep="\t")
)
a = item[item.rating == 'ņĀÉ 5.0ņĀ']
b = item[item.rating == 'м†Р 4.8м†']
tmp = list(a.index)
tmp.extend(list(b.index))

tmp2 = list(train[train.item == a.item.values[0]].index)
tmp2.extend(list(train[train.item == b.item.values[0]].index))

item.drop(index=tmp, inplace=True)
train.drop(index=tmp2, inplace=True)

train = train.groupby("house").filter(lambda x: len(x) >= 15)


# In[44]:


train.item.nunique()


# In[45]:


train = (
    pd.read_csv(os.path.join(data_path, "train.tsv"), sep="\t")
)

train.item.nunique()


# In[46]:


train.to_csv("../backend/data/train_.tsv", sep="\t", index=False)


# In[47]:


train = (
    pd.read_csv(os.path.join(data_path, "train.tsv"), sep="\t")
)

train = train.groupby("house").filter(lambda x: len(x) >= 15)

train.item.nunique()


# In[48]:


train.to_csv("../backend/data/train_.tsv", sep="\t", index=False)


# In[3]:


item.sort_values("title", inplace=True)


# In[9]:


print(item[10003:100010].title)


# In[ ]:




