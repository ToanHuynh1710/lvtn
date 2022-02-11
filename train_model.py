import pandas as pd
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import os

#data = pd.read_csv('./data/group_data/group_questions.csv')
stop_word = pd.read_csv("./stop_words.csv",header=None)
list_stop_word = [word for word in stop_word[0]]
path = os.listdir("./data")

def tokenizer(row):
    return word_tokenize(row, format="text")   


def drop_stop_word(data):
    data["Text"] = data.Text.apply(tokenizer)
    data["Text"] = [i.lower().split() for i in data["Text"]]
    list_ques =[]
    for row in data["Text"]:
        new_ques = [sentense for sentense in row if sentense not in list_stop_word]
        ques = [" ".join(new_ques)]
        list_ques.append(ques)
    df_list_ques = pd.DataFrame(list_ques)    
    return df_list_ques[0]

def fit_emb_model(data, file_name):
    emb = TfidfVectorizer()
    emb.fit(data)
    with open("./model/emb_"+file_name+".pkl",'wb') as outfile:
        pickle.dump(emb,outfile)
        print("emb_"+file_name+" saved")
    return emb    
def embedding(X_train, emb):
    X_train =  emb.transform(X_train)
    return X_train

def fit_model(X_train,y_train, emb, file_name):
    y_train = y_train
    class_names = sorted(list(set(y_train)))
    print("class name:",class_names)
    X_train = embedding(X_train,emb)
    #model = RandomForestClassifier(n_estimators=400, max_features="log2", criterion="gini", bootstrap=True, random_state=42)
    model = KNeighborsClassifier(n_neighbors=2, weights="distance", algorithm = 'brute',  metric = "minkowski", p = 2)
    model.fit(X_train,y_train)
    print("class name model", model.classes_)
    with open("./model/knn_model_"+file_name+".pkl",'wb') as outfile_svm:
        pickle.dump((model,class_names), outfile_svm)
    print("svm_model_"+file_name+" saved")

# X_train= df_list_ques[0]
# y_train = data["Sentiment"]

print(path)
for p in path:
    file_train = p
    file_name = p.split(".")[0]
    data = pd.read_csv('./data/'+file_train)
    X_train = drop_stop_word(data)
    y_train = data["Sentiment"]
    emb = fit_emb_model(X_train, file_name)
    fit_model(X_train, y_train, emb, file_name)
    
