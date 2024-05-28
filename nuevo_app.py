from flask import Flask, request, render_template, jsonify
from flask_executor import Executor
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import random
import re
import unicodedata
from sklearn.preprocessing import LabelEncoder
import logging
import time

app = Flask(__name__)
executor = Executor(app)

# Configuración de logs
logging.basicConfig(level=logging.DEBUG)

# Descargar los recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('wordnet')

# Cargar el modelo y otros datos necesarios
model = load_model('chatbot_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with open('tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
    tokenizer = Tokenizer()
    tokenizer.word_index = tokenizer_data

with open('label_encoder.json', 'r') as f:
    label_encoder_data = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_data)

with open('word_index.json', 'r') as f:
    word_index = json.load(f)
embedding_matrix = np.load('embedding_matrix.npy')

max_length = 20

lemmatizer = WordNetLemmatizer()

with open('data.json', encoding='utf-8') as f:
    intents = json.load(f)

def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def tokenize_and_lemmatize(text):
    text = normalize_text(text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return ' '.join(lemmatized_tokens)

def intent_detection(user_input):
    logging.debug("Starting intent detection")
    start_time = time.time()
    tokenized_input = tokenize_and_lemmatize(user_input)
    input_seq = tokenizer.texts_to_sequences([tokenized_input])
    input_seq = pad_sequences(input_seq, maxlen=max_length)
    logging.debug("Input sequence prepared")
    prediction = model.predict(input_seq)
    logging.debug("Model prediction completed")
    encoded_response = np.argmax(prediction)
    response = label_encoder.inverse_transform([encoded_response])[0]
    score = prediction[0][encoded_response]
    end_time = time.time()
    logging.debug(f"Response: {response}, Score: {score}")
    logging.debug(f"Intent detection time: {end_time - start_time} seconds")
    return response, score

def remove_accents_and_symbols(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

palabras_bien = ["bien", "super", "lo maximo", "excelente", "bendecido", "genial", "fantastico", "maravilloso", "increible", "estupendo", "fenomenal"]

palabras_mal = ["mal", "regular", "no me siento bien", "no estoy bien", "desanimado", "decaido", "apagado", "mas o menos", "bajo de nota", "triste", "abatido", "preocupado", "desalentado", "cansado", "agobiado"]

respuestas_saludo = ["Hola, ¿cómo estás?", "¡Hola! ¿Cómo te va?", "Saludos, ¿cómo te encuentras?", "Hola, ¿cómo has estado?", "¿Qué tal estás hoy?", "¿Cómo te sientes?", "Hola, ¿cómo va todo?", "¿Cómo te encuentras?", "¿Cómo va tu día?", "Hola, ¿cómo te sientes hoy?", "¿Cómo andas?"]

respuestas_bien = ["Me alegra escuchar eso. ¿En qué puedo ayudarte?", "¡Genial! ¿Cómo puedo asistirte?", "¡Qué bueno! ¿En qué te puedo ayudar?", "Eso suena fantástico. ¿Hay algo específico en lo que pueda colaborar?", "Qué bueno saberlo. ¿Necesitas algún tipo de ayuda o apoyo adicional?", "Me alegra escuchar eso. ¿Hay algo en lo que pueda contribuir para que tu día sea aún mejor?", "Maravilloso. ¿Hay algo en lo que necesites ayuda o asistencia?", "Me alegra mucho escuchar eso. ¿Cómo puedo ayudarte hoy?", "Excelente. ¿Hay algo en lo que necesites que te ayude o apoye?", "Me alegra saberlo. ¿En qué puedo colaborar para que sigas sintiéndote así de bien?"]

respuestas_mal = ["Lamento escuchar eso. ¿En qué puedo ayudarte?", 
                "Lo siento, ¿cómo puedo asistirte?", 
                "Qué pena, ¿en qué te puedo ayudar?",
                "Oh, eso no suena bien. ¿Puedo hacer algo por ti?",
                "Me entristece escuchar eso. ¿Hay algo en lo que pueda colaborar?",
                "Lo siento mucho. ¿Hay algo que pueda hacer para ayudar?",
                "Qué lástima. ¿Hay algo en lo que pueda ser de ayuda?",
                "Vaya, eso no suena nada bien. ¿Necesitas algún tipo de apoyo?",
                "Oh, lo siento mucho. ¿Hay algo que pueda hacer para hacerte sentir mejor?",
                "Lo siento mucho por eso. ¿Puedo hacer algo para ayudar a mejorar tu día?",
                "Qué pena escuchar eso. ¿Cómo puedo brindarte apoyo en este momento?"]

def respond_to_user(user_input, intents):
    user_input_lower = user_input.lower()
    user_input_clean = remove_accents_and_symbols(user_input_lower)

    if "hola" in user_input_clean:
        return random.choice(respuestas_saludo), "saludo", 1.0

    if user_input_clean == "si":
        return "¿En qué puedo ayudarte?", "respuesta_si", 1.0
    elif user_input_clean == "no":
        return "Espero verte pronto.", "respuesta_no", 1.0

    if any(palabra in user_input_clean for palabra in palabras_bien) and not any(neg_word in user_input_clean for neg_word in palabras_mal):
        return random.choice(respuestas_bien), "bien", 1.0
    elif any(neg_word in user_input_clean for neg_word in palabras_mal):
        return random.choice(respuestas_mal), "mal", 1.0

    response, score = intent_detection(user_input)
    threshold = 0.9
    if score < threshold:
        return 'No entiendo. ¿Puedes reformular la pregunta?', "confusion", score

    for intent in intents['intents']:
        if intent['tag'] == response:
            return random.choice(intent['responses']), response, score

    return "Lo siento, no entiendo lo que dices.", "unknown", 1.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    future = executor.submit(respond_to_user, user_input, intents)
    response, intent, score = future.result()
    return jsonify({'response': response, 'intent': intent, 'score': score})

if __name__ == '__main__':
    app.run(debug=True)
