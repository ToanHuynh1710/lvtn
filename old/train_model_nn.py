import gensim, re
import numpy as np
import pandas as pd
import pickle
from os import listdir
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import sys
import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
def tokenizer(row):
    return word_tokenize(row, format="text")

def txtTokenizer(texts):
    tokenizer = Tokenizer()
    # fit the tokenizer on our text
    tokenizer.fit_on_texts(texts)

    # get all words that the tokenizer knows
    word_index = tokenizer.word_index
    return tokenizer, word_index

def preProcess(sentences):

    text = [re.sub(r'([^\s\w]|)+', '', sentence) for sentence in sentences if sentence!='']
    text = [sentence.lower().strip().split() for sentence in text]
    #print("Tex=",text)
    return text
texts = []
data = pd.read_csv('./new_data_ques.csv')
data["Text"] = data.Text.apply(tokenizer)

data_raw, labels = data["Text"],data["Sentiment"]
texts = [i for i in data_raw]
texts = preProcess(texts)

tokenizer, word_index = txtTokenizer(texts)

# put the tokens in a matrix
X = tokenizer.texts_to_sequences(texts)

X = pad_sequences(X)

# prepare the labels
y = pd.get_dummies(labels)

#print((X[10:30]))
#print((y[10:30]))
#print((texts[10:30]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
word_model = gensim.models.Word2Vec(texts,min_count=1,
                     window=2,
                     size=300,
                     sample=6e-5,
                     min_alpha=0.0007, 
                     negative=20)
#word_model.save(data_folder + sep + "word_model.save")

print(word_model)
#print(word_model.wv.most_similar('xét'))
embedding_matrix = np.zeros((len(word_model.wv.vocab) + 1, 300))
for i, vec in enumerate(word_model.wv.vectors):
	embedding_matrix[i] = vec

def scheduler(epoch, lr):
    # Nếu dưới 5 epoch
    if epoch < 5:
        # Trả về lr
        return float(lr)
    else:
        # Còn không thì trả về
        return float(lr * tf.math.exp(-0.1))

filepath="checkpoint.hdf5"
callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
callback_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
callback_learning = tf.keras.callbacks.LearningRateScheduler(scheduler)

model = Sequential()
model.add(Embedding(len(word_model.wv.vocab)+1,300,input_length=X.shape[1],weights=[embedding_matrix],trainable=False))
model.add(LSTM(300,return_sequences=False))
model.add(Dense(y.shape[1],activation="softmax"))
model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])

batch = 4
epochs = 100
model.fit(X_train,y_train,batch,epochs,callbacks=[callback_earlystop,callback_checkpoint,callback_learning])
model.save("predict_model.save")
score = model.evaluate(X_test,y_test)
