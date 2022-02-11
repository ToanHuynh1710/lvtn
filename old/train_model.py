import pandas as pd
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

data = pd.read_csv('./data/new_data_ques.csv')
stop_word = pd.read_csv("./data/stop_words.csv",header=None)
list_stop_word = [word for word in stop_word[0]]


def tokenizer(row):
    return word_tokenize(row, format="text")   


data["Text"] = data.Text.apply(tokenizer)
data["Text"] = [i.lower().split() for i in data["Text"]]
list_ques =[]

for row in data["Text"]:
    new_ques = [sentense for sentense in row if sentense not in list_stop_word]
    ques = [" ".join(new_ques)]
    list_ques.append(ques)
df_list_ques = pd.DataFrame(list_ques)
X_train= df_list_ques[0]
y_train = data["Sentiment"]
global emb
emb = TfidfVectorizer()
emb.fit(X_train)
with open("./model/emb_model.pkl",'wb') as outfile:
	pickle.dump(emb,outfile)
	print("emb_model saved")
def embedding(X_train):
    X_train =  emb.transform(X_train)
    return X_train


print(X_train)
X_train = embedding(X_train)
class_names = list(set(y_train))
print(class_names)
model = SVC(kernel="rbf",C=10000,probability=True,gamma=0.0001)
model.fit(X_train,y_train)
with open("./model/svm_model.pkl",'wb') as outfile_svm:
	pickle.dump((model,class_names), outfile_svm)
	print("svm_model saved")