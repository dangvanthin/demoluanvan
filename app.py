from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
import os,string,re,operator
from keras_radam import RAdam
from pyvi import ViTokenizer, ViPosTagger
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


global graph
graph = tf.get_default_graph() 

global model

model_res = load_model('pickle/Res_model.h5',custom_objects={'RAdam': RAdam})
maxLength_res = 91

model_hotel = load_model('pickle/Hotel_model.h5',custom_objects={'RAdam': RAdam})


with open('pickle/input_tokenizer_res.pkl', 'rb') as fp:
    input_tokenizer_res = pickle.load(fp)

with open('pickle/input_tokenizer_hotel.pkl', 'rb') as fp:
    input_tokenizer_hotel = pickle.load(fp)
    
    
# Các bước tiền xử lý văn bản
def normalText(sent):
    sent = str(sent).replace('_',' ').replace('/',' trên ')
    sent = re.sub('-{2,}','',sent)
    sent = re.sub('\\s+',' ', sent)
    patPrice = r'([0-9]+k?(\s?-\s?)[0-9]+\s?(k|K))|([0-9]+(.|,)?[0-9]+\s?(triệu|ngàn|trăm|k|K|))|([0-9]+(.[0-9]+)?Ä‘)|([0-9]+k)'
    patHagTag = r'#\s?[aăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ]+'
    patURL = r"(?:http://|www.)[^\"]+"
    sent = re.sub(patURL,'website',sent)
    sent = re.sub(patHagTag,' hagtag ',sent)
    sent = re.sub(patPrice, ' giá_tiền ', sent)
    sent = re.sub('\.+','.',sent)
    sent = re.sub('(hagtag\\s+)+',' hagtag ',sent)
    sent = re.sub('\\s+',' ',sent)
    return sent


def normalize_elonge_word(sent):
    s_new = ''
    for word in sent.split(' '):
        word_new = ''
        for char in word.strip():
            if char != word_new[-1]:
                word_new += char
    s_new += word_new.strip() + ' '
    return s_new

def tokenizer(text):
    token = ViTokenizer.tokenize(text)
    token = token.replace('giá tiền','giá_tiền').replace('Giá tiền','Giá_tiền')
    return token

def deleteIcon(text):
    text = text.lower()
    s = ''
    pattern = r"[a-zA-ZaăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ,._]"
    
    for char in text:
        if char !=' ':
            if len(re.findall(pattern, char)) != 0:
                s+=char
            elif char == '_':
                s+=char
        else:
            s+=char
    s = re.sub('\\s+',' ',s)
    return s.strip()

def normalize_elonge_word(sent):
    s_new = ''
    for word in sent.split(' '):
        word_new = ' '
        for char in word.strip():
            if char != word_new[-1]:
                word_new += char
        s_new += word_new.strip() + ' '
    return s_new.strip()  

def clean_doc(doc):
    for punc in string.punctuation:
        doc = doc.replace(punc,' '+ punc + ' ')
    doc = normalText(doc)
    doc = deleteIcon(doc)
    doc = tokenizer(doc)
    # Lowercase
    doc = doc.lower()
    # Removing multiple whitespaces
    doc = re.sub(r"\?", " \? ", doc)
    # Remove numbers
    doc = re.sub(r"[0-9]+", " num ", doc)
    # Split in tokens
    # Remove punctuation
    for punc in string.punctuation:
        if punc not in "_":
            doc = doc.replace(punc,' ')
    doc = re.sub('\\s+',' ',doc)
    doc = normalize_elonge_word(doc)
    return doc

@app.route("/")
def home():
    return render_template("restaurant.html")


@app.route("/dulieu")
def dulieu():
    return render_template("dulieu.html")


@app.route("/phantichketqua")
def phantichketqua():
    return render_template("phantichketqua.html")


@app.route("/hoteldomain/")
def hoteldomain():
    return render_template("hotel.html")


maxLength_hotel = 142 
listLabel_hotel = 'HOTEL#GENERAL,ROOMS#GENERAL,ROOM_AMENITIES#GENERAL,FACILITIES#GENERAL,SERVICE#GENERAL,LOCATION#GENERAL,HOTEL#PRICES,ROOMS#PRICES,ROOM_AMENITIES#PRICES,FACILITIES#PRICES,FOOD&DRINKS#PRICES,HOTEL#DESIGN&FEATURES,ROOMS#DESIGN&FEATURES,ROOM_AMENITIES#DESIGN&FEATURES,FACILITIES#DESIGN&FEATURES,HOTEL#CLEANLINESS,ROOMS#CLEANLINESS,ROOM_AMENITIES#CLEANLINESS,FACILITIES#CLEANLINESS,HOTEL#COMFORT,ROOMS#COMFORT,ROOM_AMENITIES#COMFORT,FACILITIES#COMFORT,HOTEL#QUALITY,ROOMS#QUALITY,ROOM_AMENITIES#QUALITY,FACILITIES#QUALITY,FOOD&DRINKS#QUALITY,FOOD&DRINKS#STYLE&OPTIONS,HOTEL#MISCELLANEOUS,ROOMS#MISCELLANEOUS,ROOM_AMENITIES#MISCELLANEOUS,FACILITIES#MISCELLANEOUS,FOOD&DRINKS#MISCELLANEOUS'
categories_hotel = listLabel_hotel.split(',')



listLabel_res = 'FOOD#STYLE&OPTIONS,FOOD#PRICES,FOOD#QUALITY,DRINKS#STYLE&OPTIONS,DRINKS#QUALITY,DRINKS#PRICES,RESTAURANT#GENERAL,RESTAURANT#MISCELLANEOUS,RESTAURANT#PRICES,LOCATION#GENERAL,SERVICE#GENERAL,AMBIENCE#GENERAL'
categories_res = listLabel_res.split(',')
@app.route("/restaurantanalysis/", methods=['POST','GET'])
def restaurantanalysis():
    query_value = request.form['query']
    cleaned_query = clean_doc(query_value)
    textArray_test = np.array(pad_sequences(input_tokenizer_res.texts_to_sequences([cleaned_query]), maxlen=maxLength_res,padding="post"))
    s = ''
    for iz,item in enumerate(textArray_test):
        with graph.as_default():
            predicted = model_res.predict([np.expand_dims(item, axis=0)])
            for i, predict in enumerate(predicted):
                index2, value = max(enumerate(predict[0]), key=operator.itemgetter(1))
                if index2 == 1:
                    s+= '{' + str(categories_res[i]) + ', positive}, '
                elif index2 == 2:
                    s+= '{' + str(categories_res[i]) + ', neutral}, '
                elif index2 == 3:
                    s+= '{' + str(categories_res[i]) + ', negative}, '
    label  = s[:-2]
    return render_template("restaurant_result.html", data = [{"label":label,"query":query_value}])

@app.route("/hotelanalysis/", methods=['POST','GET'])
def hotelanalysis():
    query_value = request.form['query']
    cleaned_query = clean_doc(query_value)
    textArray_test = np.array(pad_sequences(input_tokenizer_hotel.texts_to_sequences([cleaned_query]), maxlen=maxLength_hotel,padding="post"))
    s = ''
    for iz,item in enumerate(textArray_test):
        with graph.as_default():
            predicted = model_hotel.predict([np.expand_dims(item, axis=0)])
            for i, predict in enumerate(predicted):
                index2, value = max(enumerate(predict[0]), key=operator.itemgetter(1))
                if index2 == 1:
                    s+= '{' + str(categories_hotel[i]) + ', positive}, '
                elif index2 == 2:
                    s+= '{' + str(categories_hotel[i]) + ', neutral}, '
                elif index2 == 3:
                    s+= '{' + str(categories_hotel[i]) + ', negative}, '
    label  = s[:-2]
    return render_template("hotel_result.html", data = [{"label":label,"query":query_value}])

if __name__ == "__main__":
    app.run(debug=True)
