import os
import nltk
import numpy
import random
import pyttsx3
import tflearn
import datetime
import tensorflow
import speech_recognition as sr
from tensorflow.python.framework import ops
from nltk.stem.lancaster import LancasterStemmer
from load import model, labels, words, data
stemmer = LancasterStemmer()
ArchMagos = pyttsx3.init()
voice = ArchMagos.getProperty('voices')
# id 63=VN
ArchMagos.setProperty('voice', voice[16].id)
ArchMagos.setProperty('rate',160)
model.load("model.tflearn")

def speak(audio):
    print('F.R.I.D.A.Y :' + audio)
    ArchMagos.say(audio)
    ArchMagos.runAndWait()

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("Master: ")
        if inp.lower() == "quit":
            speak("system is shuting down. see you later Master")
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        res = random.choice(responses)
        speak(res)


chat()