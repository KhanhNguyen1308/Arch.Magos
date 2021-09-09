import cv2
import json
import pyttsx3
import wikipedia
import urllib.request
import speech_recognition as sr
from queue import Queue
from threading import Thread
from yolo import video_capture
from tensorflow.keras.models import load_model
from function import count_object, predict_class, get_response
wikipedia.set_lang("en")
born_time=1629699876.6019154
intents = json.loads(open("cfg/Arch_Master.json").read())
Arch = pyttsx3.init('espeak')
voice = Arch.getProperty('voices')
Arch.setProperty('voice', voice[16])
Arch.setProperty('rate',150)
show_on = False  
show_off = False 
resize = False

def speak(audio):
    print('Arch:' + audio)
    Arch.say(audio)
    Arch.runAndWait()

work_list = ["Youtube", "Google", "Wikipedia", "music"]
print('Arch is ready master')
while True:
    message = input("Master: ")
    ints = predict_class(message)
    tag = ints[0]['intent']
    res = get_response(ints, intents)
    if tag not in work_list:
        speak(res)
    if tag == "quit":
        break
    if tag == "Youtube":
        res = "Yes my Lord! What're you looking for?"
        speak(res)
        search = input("Master: ")
        url = urllib.request.urlopen("https://www.youtube.com/results?search_query="+search)
        speak("Here is some results i found for "+search)
    if tag == "Google":
        res = "Yes my Lord! What're you looking for?"
        speak(res)
        search = input("Master: ")
    if tag == "Wikipedia":
        res = "Your Highness! What're you looking for?"
        speak(res)
        search = input("Master: ")
        try:
            result = wikipedia.summary(search, sentences=3)
            speak(result)
        except Exception:
            speak("Please give me more information, my Lord!")
    Thread(target=video_capture, args=(show_on, show_off, resize)).start()
