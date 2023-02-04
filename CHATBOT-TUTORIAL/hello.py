from flask import Flask
from flask import request
from flask import Flask, render_template
import random
import json
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
import math
import re
from collections import Counter
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from nltk.tokenize import word_tokenize

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cosine_similarity_strings(s1, s2):
    print(s1)
    s1 = torch.tensor(s1)
    s2 = torch.tensor(s2)
    similarity = cosine_similarity(s1, s2)
    return similarity

with open('intents.json', 'r') as f:
    intents = json.load(f)

@app.route('/')
def home():
   return render_template('index.html')

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"] 

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
# conda activate my-torch
bot_name = "Sam"


WORD = re.compile(r"\w+")


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

@app.route("/get-keyword")
def hello_world():
    args = request.args
    prompt=args['prompt']
    print(args)
    
    while True:
        sentence = 'Which items do you have?'
        if sentence == "quit":
            break

        sentence = tokenize(prompt)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _,predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output,dim=1)
        probs = probs[0] [predicted.item()]

        best_response = ""
        min_similarity = 10

        if probs.item() > 0.75 :

            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    similarity = -10
                    for response in intent["responses"]:
                        sentence = tokenize(response)
                        X = bag_of_words(sentence, all_words)
                        X = X.reshape(1, X.shape[0])
                        X = torch.from_numpy(X)

                        output = model(X)
                        _,predicted = torch.max(output, dim=1)
                        tag = tags[predicted.item()]

                        probs = torch.softmax(output,dim=1)
                        similarity = probs[0] [predicted.item()]
                        print(response)
                        print(similarity.item())
                        if similarity < min_similarity:
                            min_similarity = similarity
                            best_response = response
                    print("------------")
                    print(best_response)
                    return {'result': random.choice(intent['responses'])}
        else:
            # print("2")
            return {'result': 'I dont understand...'}
            
    

