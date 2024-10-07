import numpy as np
import tensorflow as tf
import tflearn
import random
import json
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Database connection setup function
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="chatbot_db",
        user="postgres",
        password="root"
    )

# Download the tokenizer if not already done
nltk.download('punkt')

stemmer = LancasterStemmer()

# Load intents file
with open('intents.json', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Initialize lists for processing intents
words = []
classes = []
documents = []
ignore_words = ['?']

# Process intents to convert them into words, classes, and documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 512, activation='relu', regularizer='L2', weight_decay=0.001)
net = tflearn.batch_normalization(net)
net = tflearn.dropout(net, 0.6)
net = tflearn.fully_connected(net, 256, activation='relu', regularizer='L2', weight_decay=0.001)
net = tflearn.batch_normalization(net)
net = tflearn.dropout(net, 0.6)
net = tflearn.fully_connected(net, 128, activation='relu', regularizer='L2', weight_decay=0.001)
net = tflearn.batch_normalization(net)
net = tflearn.dropout(net, 0.6)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')

initial_learning_rate = 0.0005
net = tflearn.regression(net, optimizer='adam', learning_rate=initial_learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Check if the model already exists before training
if os.path.exists('model.tflearn.index'):
    model.load('model.tflearn')
    print("Model loaded from file.")
else:
    model.fit(train_x, train_y, n_epoch=2000, batch_size=16, show_metric=True)
    model.save('model.tflearn')
    print("Model trained and saved to file.")
    pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

# Load training data
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

def save_chat(user_id, user_input, bot_response):
    # Ensure the database connection is active and create a new cursor
    conn = get_db_connection()
    try:
        query = "INSERT INTO chat_history (user_id, user_input, bot_response) VALUES (%s, %s, %s)"
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id, user_input, bot_response))
            conn.commit()
    finally:
        conn.close()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    bot_response = random.choice(i['responses'])
                    save_chat(userID, sentence, bot_response)
                    return bot_response
            results.pop(0)

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json['message']
    answer = response(user_input)
    return jsonify(answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
