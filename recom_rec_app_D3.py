# not json format trying with the css top 5 recommendation 
from flask import Flask, jsonify, request, render_template
from flask_jsonpify import jsonpify
import json, requests, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  

#from ingredient_parser import ingredient_parser
import rec_sys1_D3

from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import nltk
import string
import re
import unidecode, ast

import tensorflow 
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.utils import load_img, img_to_array 
import config

from keras.applications.inception_v3 import preprocess_input
model = load_model(config.MultiClass1)
mode2 = load_model(config.UniClass)
model.make_predict_function()

global k
k = 0.1
###### Functions
def parser(ingredients):
    words_to_remove = ["ADVERTISEMENT","advertisement", "advertisements"]

    ingredients = ingredients.split()

    resultwords  = [word for word in ingredients if word.lower() not in words_to_remove]
    resultwords = ' '.join(resultwords) 
    return resultwords

LABELS =['Apple', 'Cucumber', 'Kiwi', 'Onion', 'Pepper', 'Strawberry', 'Tomato']

def covert_onehot_string_labels(label_string,label_onehot):
  labels=[]
  for i, label in  enumerate(label_string):
     if label_onehot[i]:
       labels.append(label)
  if len(labels)==0:
    labels.append("NONE")
  return labels

def predict_class(img_path):
    #k = 0.1
    i= image.load_img(img_path, target_size=(299, 299)) #224 M3 # 299 M1/2
    img = image.img_to_array(i)
    img =  np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    pred = model.predict(img)
    pred = np.array(pred)
    print("predicted: " ,pred)
    pred = np.around(pred,1)
    #print("arrounded: " ,pred)
    pred[pred>=k]=1
    pred[pred<k]=0
    #print("pred: " ,pred)
    pred = list(pred[0])

    #tit = str(covert_onehot_string_labels(LABELS, pred))
    tit = covert_onehot_string_labels(LABELS, pred)
    print("predicted: ",pred, tit)
    return(tit)

 ###### routes   
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("Recipe2.html")

@app.route('/display', methods =["GET", "POST"])
def display():
    if request.method == "POST":
        global src,R_name,R_ingred,R_inst
        ID = request.form['submit_button'] 
        #R_num = request.args.get('button_ID')
        if request.form['submit_button'] == '0':
            R_name = recipes[0]
            R_ingred = ingredients[0] #parser(ingredients[0])
            R_inst = instructions[0]
            img_URL= R_img[0]
            src = 'static/11.jpg'            
            #src= 'static/0.jpg' 11.jpg              
        elif request.form['submit_button'] == '1':
            R_name = recipes[1]
            R_ingred = ingredients[1] #parser(ingredients[1])
            R_inst = instructions[1] 
            img_URL= R_img[1] 
            src= "static/12.jpg" 
            #src= 'static/1.jpg'  
        elif request.form['submit_button'] == '2':
            R_name = recipes[2]
            R_ingred = ingredients[2] #parser(ingredients[2])
            R_inst = instructions[2] 
            img_URL= R_img[2]
            src= "static/13.jpg" 
            #src= 'static/2.jpg'  
        elif request.form['submit_button'] == '3':
            R_name = recipes[3]
            R_ingred = ingredients[3] #parser(ingredients[3])
            R_inst = instructions[3]  
            img_URL= R_img[3]
            src= 'static/14.jpg' 
            #src= 'static/3.jpg'  
        elif request.form['submit_button'] == '4':
            R_name = recipes[4]
            R_ingred = ingredients[4] #parser(ingredients[4])
            R_inst = instructions[4] 
            img_URL= R_img[4]
            src= 'static/15.jpg' 
            #src= 'static/4.jpg' 

        #R_ingred = parser(R_ingred)  
        #R_ingred.replace('ADVERTISEMENT', '')   
           
    return render_template("display.html",src = src, recipe_name = R_name,recipes_ingredients = R_ingred,recipes_instructions = R_inst)



@app.route('/recom', methods =["GET", "POST"])
def recommend_recipe():
    if request.method == "POST":
        global ing,recipes,ingredients,instructions,R_img
        if request.form.get('ingrec') == 'value1':
            ing = request.form.get("ing")
        elif request.form.get('sub') == 'value2':
            #get_output()
            img = request.files['my_image']
            img_path = "static/" + img.filename	
            img.save(img_path)
            global p
            p = predict_class(img_path) #predict_label

            return render_template("Recipe2.html", prediction = p, img_path = img_path)

        elif request.form.get('imrecom') == 'value3':
            print(p)
            #print(type(p))
            ing = ','.join([str(elem) for elem in p]) 
            #ing = ",".join(p) 
            #print(ing)
        recipe = rec_sys1_D3.get_recs(str(ing))
        print("TFID Recommendation : ",recipe.score)
        print(recipe)
        recipes = []
        ingredients = []
        instructions = []
        R_img =[] #### D2 image scrapped 

        for i in  range (len(recipe)):
            recipes.append(recipe.iloc[i][0])
            ingredients.append(recipe.iloc[i][1])
            instructions.append(recipe.iloc[i][3])
            R_img.append(recipe.iloc[i][4])

        for i in range(len(R_img)):
            img_URL= R_img[i]
            src = 'static/'+'1'+f'{i+1}'+'.jpg'
            img_data = requests.get(img_URL).content
            with open(src,'wb') as handler:
                handler.write(img_data)         
    return render_template("recom.html",ingredients = str(ing),recipe_name = recipes) ## R_img

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)