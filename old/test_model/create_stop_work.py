import pandas as pd
import collections
from underthesea import word_tokenize
text = pd.read_csv("../data/new_data_ques.csv")
stop_word = pd.read_csv("../data/stop_words.csv",header=None)
list_stop_word = [word for word in stop_word[0]]
def tokenizer(row):
    return word_tokenize(row, format="text")
#dem so tu`
text["Text"] = text.Text.apply(tokenizer)
text["Text"] = [i.lower().split() for i in text["Text"]]
# data = list(text["Text"])
# new_data =[]
# for i in data:
#     for j in i:
#         new_data.append(j)
# arr_txt = collections.Counter(new_data)
# print(arr_txt)
# print(arr_txt['việc_làm'])
list_ques =[]
print(list_stop_word)
for row in text["Text"]:
	new_ques = [sentense for sentense in row if sentense not in list_stop_word]
	ques = [" ".join(new_ques)]
	list_ques.append(ques)
df_list_ques = pd.DataFrame(list_ques)
print(df_list_ques)

