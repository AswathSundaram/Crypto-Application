import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs except errors
from tensorflow.keras.models import load_model

import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure the file paths are correct
model_path = r'D:\Others\Demo-Works\Crypto-Chatbot-master\Backend\krypto_model.h5'
intents_path = r'D:\Others\Demo-Works\Crypto-Chatbot-master\Backend\intents.json'
words_path = r'D:\Others\Demo-Works\Crypto-Chatbot-master\Backend\words.pkl'
classes_path = r'D:\Others\Demo-Works\Crypto-Chatbot-master\Backend\classes.pkl'

# Load the model and other data files
model = load_model(model_path)

with open(intents_path, 'r') as file:
    intents = json.load(file)

words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower(), pos='v') for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    response = getResponse(ints, intents)
    return {"answer": response}

# For testing the chatbot_response function
if __name__ == "__main__":
    print(chatbot_response("Hello"))





