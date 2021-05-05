from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from college_management_app.EmailBackend import EmailBackend
from django.contrib import messages
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt


def showDemoPage(request):
    return render(request,"demo.html")

def ShowLoginPage(request):
    return render(request,"login_page.html")

def doLogin(request):
    if request.method!="POST":
        return HttpResponse("<h2>Method Not Allowed</h2>")
    else:
        user=EmailBackend.authenticate(request,username=request.POST.get("email"),password=request.POST.get("password"))
        if user!=None:
            login(request,user)
            if user.user_type=="1":
                return HttpResponseRedirect('/admin_home')
            elif user.user_type=="2":
                return HttpResponseRedirect(reverse("teacher_home"))
            else:
                return HttpResponseRedirect(reverse("student_home"))

        else:
            messages.error(request,"Invalid Login Details")
            return HttpResponseRedirect("/")

def GetUserDetails(request):
    if request.user!=None:
        return HttpResponse("User : "+request.user.email+" usertype : "+request.user.user_type)
    else:
        return HttpResponse("Please Login First")

def logout_user(request):
    logout(request)
    return HttpResponseRedirect("/")


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbotmodel.h5')
import json
import random
from django.shortcuts import render

DATA_PATH = 'intents.json'
intents = json.loads(open(DATA_PATH,encoding="utf8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):

    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence):
    res = model.predict(np.array([bow(sentence, words)]))[0]
    ERROR_THRESHOLD = 0.50



    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)



    return_list = list()
    for r in results:
        rr = [classes[r[0]]],[r[1]]
        return_list.extend(rr)
    return return_list

def chat(request):
    msg = request.POST.get("msg")
    results = predict_class(sentence=msg)

    results_index = np.array(results)
    confidence = results_index[1]
    co = (confidence.astype('float64'))
    val = np.float32(co)
    pyval = val.item()

    if pyval > 0.6:
        tag = results_index[0]

        list_of_intents = intents['intents']
        for i in list_of_intents:
            if (i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return HttpResponse(result)
    else:
        return HttpResponse('Sorry, I did not understand that')
    

