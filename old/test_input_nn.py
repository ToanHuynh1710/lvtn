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
data = input("nhap cau hoi:")
data = [tokenizer(data)]
print(data)

texts = [i for i in data]
texts = preProcess(texts)
print(texts)


tokenizer, word_index = txtTokenizer(texts)


X = tokenizer.texts_to_sequences(texts)
model = load_model("predict_model.save")
model.summary()
y_pred = model.predict(X)
print(y_pred)