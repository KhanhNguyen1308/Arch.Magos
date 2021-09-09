import cv2
import nltk
import json
import random
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("cfg/Arch_Master.json").read())
words = pickle.load(open('cfg/words.pkl', 'rb'))
classes = pickle.load(open('cfg/classes.pkl', 'rb'))
model = load_model('model/Arch.h5')


def count_object(classes):
    face, person = 0, 0
    for object in classes:
        if object[0]== 0:
            face+=1
        if object[0]== 1:
            person+=1
    return face, person



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_word(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in  sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_word(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_thres = 0.25
    results = [[i, r] for i, r in enumerate(res) if r> ERROR_thres]
    results.sort(key=lambda x : x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability':str(r[1])})
    return return_list


def get_response(intent_list, intent_json):
    tag = intent_list[0]['intent']
    list_of_intents = intent_json['intents']
    for i in list_of_intents:
        if i['tag']== tag:
            result = random.choice(i['responses'])
            break
    return result
