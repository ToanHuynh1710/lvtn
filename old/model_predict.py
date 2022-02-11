import speech_recognition as sr
from gtts import gTTS

import os
import pickle
import numpy as np
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd
import time


data_answer = pd.read_csv("./data/new_data_ans.csv")


def speech_to_text():
	r = sr.Recognizer()
	question_text = ""
	with sr.Microphone() as source:
		#r.pasue_threshold = 1
		#r.energy_threshold = 3000
		#r.dynamic_energy_threshold = True
		#r.adjust_for_ambient_noise(source,duration = 1.5)
		print("listening...")
		#audio = r.listen(source,timeout=5.0)
		r.pause_threshold = 0.8
		r.enery_threshold = 3000
		r.dynamic_energy_threshold = False
		r.adjust_for_ambient_noise(source, duration=0.5)
		audio = r.listen(source, timeout=5.0,phrase_time_limit=5.0)
	try:
		question_text = r.recognize_google(audio, language="vi_VN")	
	except: 
		return "none"
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

try:
	with open("./model/emb_model.pkl","rb") as emb_file:
		emb = pickle.load(emb_file)
except:
	print("khong ton tai file emb_model")
	exit()

try:
	with open("./model/svm_model.pkl","rb") as svm_file:
		model,class_names = pickle.load(svm_file)
except:
	print("khong ton tai file svm_model")
	exit()


#answer = data_answer[data_answer["Sentiment"]==1]
os.system("cvlc ./mp3/openning.mp3")
time.sleep(0.5)
while True:
	question = ""
	question = speech_to_text()
	while question == "none":
		question = speech_to_text()
		if question != "none": break
	print(question)
	question = word_tokenize(question,format="text")
	emb_question = emb.transform([question])
	class_answer = model.predict_proba(emb_question)
	best_answer = np.argmax(class_answer,axis=1)
	best_class_probabilities = class_answer[np.arange(len(best_answer)), best_answer]
	best_class = class_names[best_answer[0]]
	print(best_class)
	print(best_class_probabilities)
	if best_class_probabilities > 0.3:
		if int(best_class) == 90: break
		text_answers = data_answer[data_answer["Sentiment"] == best_class]
		for key,answer in enumerate(text_answers["Text"]):
			print(answer)
			text_to_speech(answer,key)
			time.sleep(0.5)
			play_sound("./mp3/output"+str(key)+".mp3")
			time.sleep(0.5)
	else:
		os.system("cvlc ./mp3/thresh_hold.mp3")
		time.sleep(0.5)