from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import collections
from nltk.tokenize import word_tokenize as wt


# arr_txt = []
# text = ['có bao nhiêu phương thức xét tuyển','điểm chuẩn là bao nhiêu']

# #new_text = wt(text)
# new_text = [word_tokenize(c,format="text") for c in text]
# # print(new_text)
# vector = TfidfVectorizer()
# vector.fit(new_text)
# word_vector = vector.fit_transform(new_text)
# print(vector.get_feature_names())
# print(word_vector)
data = pd.read_csv("./data/new_data_ques.csv")
print(data.shape)