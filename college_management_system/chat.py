#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random


# In[2]:



model = load_model('chatbotmodel.h5')
data = open('intents.json', encoding='utf-8').read()
intents = json.loads(data)
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


# In[3]:


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


# In[4]:


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


# In[5]:


def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    result = model.predict(np.array([p]))[0]
#   ERROR_THRESHOLD = 0.25
    ERROR_THRESHOLD = 0.02

    results = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list, result


# In[6]:



def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        else:
            print('Sorry, I cannot understand you')
    return result


# In[7]:



def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# In[ ]:


def chat(): 
    while True: 
        inp = input("Please enter your query. \n > -") 
        if inp == "quit": 
            break 
        results = predict_class(sentence=inp) 
       # return results 
       # resultsindex = np.array(results)
        tag = results[0][0] 
        listofintents = intents['intents'] 
        for i in listofintents: 
            if (i['tag'] == tag["intent"]): 
                result = random.choice(i['responses']) 
                print(result)
                break
chat()


# In[ ]:



 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




