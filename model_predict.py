import speech_recognition as sr
from gtts import gTTS
import os
import pickle
import numpy as np
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time


data_answer = pd.read_csv("./new_data_ans.csv")
name = "USB Audio Device: - (hw:2,0)"
r = sr.Recognizer()

mic_list = sr.Microphone.list_microphone_names()
for i, microphone_name in enumerate(mic_list):
    if microphone_name == name: index = i
    print(i,microphone_name)


def speech_to_text():
	r = sr.Recognizer()
	question_text = ""
	with sr.Microphone(device_index=2) as source:
		r.pause_threshold = 0.8
		r.energy_threshold = 3000
		r.dynamic_energy_threshold = True
		r.adjust_for_ambient_noise(source,duration = 1.5)
		print("listening...")
		try:
			audio = r.listen(source,timeout=5.0,phrase_time_limit=5.0)
		except:
			return None
	try:
		question_text = r.recognize_google(audio, language="vi_VN")	
	except: 
		return None
	return question_text.lower()

def text_to_speech(answer_text,num):
	voice_mp3 = gTTS(answer_text, lang="vi",slow=False)
	voice_mp3.save("./mp3/output"+str(num)+".mp3")

def play_sound(path):
	if os.path.exists(path):
		os.system("cvlc "+path)
		os.remove(path)
	else:
		print("file khong ton tai")


def get_model(path, classifier=False):
	if classifier == False:
		try:
			with open(path, "rb") as model_file:
				model = pickle.load(model_file)
		except:
			model = None
		return model
	else:
		try:
			with open(path,"rb") as svm_file:
				model,class_names = pickle.load(svm_file)
		except:
			model, class_names = None, None
		return model, class_names
list_model = dict()
list_emb = dict()
list_class = dict()
paths = os.listdir("./model")
for path in paths:
	name = path.split(".")[0]
	if name[0:3] == "emb":
		list_emb[str(name[-1])] = get_model("./model/"+path)
	else:
		list_model[str(name[-1])], list_class[str(name[-1])] = get_model("./model/"+path, classifier=True)

for key,value in list_model.items():
	if list_model[key] == None: 
		print("khong the load list_model["+key+"]")
		exit()
	if list_emb[key] == None:
		print("khong the load list_emb["+key+"]")
		exit()
	if list_class[key] == None:
		print("khong the load list_class["+key+"]")
		exit()

emb = list_emb["s"]
model = list_model["s"]
class_names = list_class["s"]
#answer = data_answer[data_answer["Sentiment"]==1]
os.system("cvlc ./mp3/openning.mp3")
while True:
	question = ""
	question = speech_to_text()
	while question is None:
		question = speech_to_text()
		if question is not None: break
	print(question)
	if question == "dừng lại":
		os.system("cvlc ./mp3/ending.mp3")
		break
	#question = input("Nhap cau hoi:")
	question = word_tokenize(question,format="text")
	emb_question = emb.transform([question])
	class_answer = model.predict_proba(emb_question)
	best_answer = np.argmax(class_answer,axis=1)
	best_class_probabilities = class_answer[np.arange(len(best_answer)), best_answer]
	best_class = class_names[best_answer[0]]
	#print(best_class)
	#print(best_class_probabilities)
	#print(class_answer)
	if best_class_probabilities > 0.5:
		sub_emb = list_emb[str(best_class)]
		sub_model = list_model[str(best_class)]
		sub_class = list_class[str(best_class)]
		sub_emb_question = sub_emb.transform([question])
		sub_class_answer = sub_model.predict_proba(sub_emb_question)
		#print(sub_class_answer)
		sub_best_answer = np.argmax(sub_class_answer,axis=1)
		sub_best_class_probabilities = sub_class_answer[np.arange(len(sub_best_answer)), sub_best_answer]
		sub_best_class = sub_class[sub_best_answer[0]]
		#print(sub_class)
		#print(sub_best_class)
		#print(sub_best_class_probabilities)
		if int(sub_best_class) >  0.3:
			text_answers = data_answer[data_answer["Sentiment"] == sub_best_class]
			for key,answer in enumerate(text_answers["Text"]):
				print(answer)
				text_to_speech(answer,key)
				time.sleep(0.5)
				play_sound("./mp3/output"+str(key)+".mp3")
		else:
			os.system("cvlc ./mp3/thresh_hold.mp3")
	else:
		os.system("cvlc ./mp3/thresh_hold.mp3")