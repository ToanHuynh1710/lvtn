import speech_recognition as sr
import pyttsx3
import datetime 
import playsound

r = sr.Recognizer()

# mic_list = sr.Microphone.list_microphone_names()
# for i, microphone_name in enumerate(mic_list):
#     print(microphone_name)
#     if microphone_name == mic_name:
#         device_id = i
        

def voice():
    with sr.Microphone() as source:
    	text = ""
    	r.pause_threshold = 1
    	r.adjust_for_ambient_noise(source)
    	print("noi gi do")
    	audio = r.listen(source)
    try:
    	text = r.recognize_google(audio, language = "vi_VN")
    except:
    	print("............")
    	voice()
    return text    

sentense = voice()
print(sentense)
# from gtts import gTTS
# import playsound

# text = "xin chào, tôi là trợ lý ảo tư vấn tuyển sinh, tôi có thể giúp gì cho bạn"
# output= gTTS(text,lang="vi", slow=False)
# output.save("./mp3/openning.mp3")

# playsound.playsound("./mp3/openning.mp3", True)