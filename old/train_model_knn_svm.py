import pandas as pd
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
#text = 'điểm chuẩn của ngành là bao nhiêu'
#text = pd.read_csv('./data_crawler.csv')
#text = pd.read_csv('./dataset.csv')
text = pd.read_csv('./data/new_data_ques.csv')
stop_word = pd.read_csv("./data/stop_words.csv",header=None)
list_stop_word = [word for word in stop_word[0]]


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
    emb = TfidfVectorizer(min_df=0, max_df=0.8,max_features=3000,sublinear_tf=True,ngram_range=(1,4))#1,4
    emb.fit(X_train)
    X_train =  emb.transform(X_train)
    X_test = emb.transform(X_test)
    return X_train, X_test

def counter_txt(row):
    return collections.Counter(row)


text["Text"] = text.Text.apply(standardize_data)
text["Text"] = text.Text.apply(tokenizer)

text["Text"] = [i.lower().split() for i in text["Text"]]
list_ques =[]
print(list_stop_word)
for row in text["Text"]:
    new_ques = [sentense for sentense in row if sentense not in list_stop_word]
    ques = [" ".join(new_ques)]
    list_ques.append(ques)
df_list_ques = pd.DataFrame(list_ques)
print(df_list_ques)



X_train,X_test,y_train,y_test = train_test_split(df_list_ques[0],text["Sentiment"],test_size=0.2, random_state=42)





X_train,X_test = embedding(X_train,X_test)

model_knn = KNeighborsClassifier(n_neighbors = 2, algorithm = 'ball_tree', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, p = 2, weights = 'distance')
model_knn.fit(X_train,y_train)
y_pred_knn = model_knn.predict(X_test)
print("y_pred_knn",y_pred_knn)
score_acc_knn = metrics.accuracy_score(y_pred_knn,y_test)
print("score_acc_knn",score_acc_knn,model_knn.score(X_test,y_test))

model_svm = SVC(kernel='linear', C=2)
model_svm.fit(X_train,y_train)
y_pred_svm = model_svm.predict(X_test)
print("y_pred_svm",y_pred_svm)
score_acc_svm = metrics.accuracy_score(y_pred_svm,y_test)
print("score_acc_svm",score_acc_svm,model_svm.score(X_test,y_test))

# from sklearn.model_selection import GridSearchCV
# j = [1,10,100,1000,10000]
# parameter_candidates = [
#   {'C': j, 'kernel': ['rbf']},
# ]

# clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
# clf.fit(X_train, y_train)
# print('Best score:', clf.best_score_)
# print('Best C:',clf.best_estimator_.C)

