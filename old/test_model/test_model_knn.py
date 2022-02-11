import pandas as pd
from nltk.tokenize import word_tokenize as wt
from sklearn.feature_extraction.text import CountVectorizer
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import re
from sklearn import metrics
from sklearn.svm import SVC
import collections
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#text = 'điểm chuẩn của ngành là bao nhiêu'
#text = pd.read_csv('./data_crawler.csv')
#text = pd.read_csv('./dataset.csv')
text = pd.read_csv('../new_data_ques.csv')
print(text.shape)
def word_token(row):
	return wt(row)

def emb(x):
	embb = CountVectorizer(max_features=1000)
	return embb.fit_transform(x).toarray()

def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row


def tokenizer(row):
    return word_tokenize(row, format="text")


def embedding(X_train, X_test):
    global  emb
    emb = TfidfVectorizer(min_df=0, max_df=0.8,max_features=3000,sublinear_tf=True,ngram_range=(1,3))
    emb.fit(X_train)
    X_train =  emb.transform(X_train)
    X_test = emb.transform(X_test)
    return X_train, X_test



text["Text"] = text.Text.apply(standardize_data)
text["Text"] = text.Text.apply(tokenizer)




X_train,X_test,y_train,y_test = train_test_split(text["Text"],text["Sentiment"],test_size=0.2, random_state=42)

X_train, X_test = embedding(X_train,X_test)

model_knn = KNeighborsClassifier(n_neighbors = 2, algorithm = 'ball_tree', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, p = 2, weights = 'distance')
model_knn.fit(X_train,y_train)
y_pred_knn = model_knn.predict(X_test)
print("y_pred_knn",y_pred_knn)
score_acc_knn = metrics.accuracy_score(y_pred_knn,y_test)
print("score_acc_knn",score_acc_knn,model_knn.score(X_test,y_test))

start_time = time.time()
clf = KNeighborsClassifier(n_neighbors = 2, algorithm = 'ball_tree', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, p = 2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print ("Accuracy of 1NN for MNIST: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print ("Running time: %.2f (s)" % (end_time - start_time))
