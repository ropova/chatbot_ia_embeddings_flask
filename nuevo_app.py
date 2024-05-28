from gevent.pywsgi import WSGIServer
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
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import LabelEncoder
import asyncio

app = Flask(__name__)

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

# Función para detectar la intención del usuario utilizando la red neuronal
async def intent_detection(user_input):
    tokenized_input = tokenize_and_lemmatize(user_input)
    print(f"Tokenized input: {tokenized_input}")  # Debugging
    input_seq = tokenizer.texts_to_sequences([tokenized_input])
    input_seq = pad_sequences(input_seq, maxlen=max_length)
    print(f"Input sequence: {input_seq}")  # Debugging
    prediction = model.predict(input_seq)
    print(f"Prediction: {prediction}")  # Debugging
    encoded_response = np.argmax(prediction)
    response = label_encoder.inverse_transform([encoded_response])[0]
    score = prediction[0][encoded_response]
    return [response], [score]

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

    # Manejar el saludo inicial
    if "hola" in user_input_clean:
        return random.choice(respuestas_saludo), "saludo", 1.0

    # Verificar si el usuario respondió "sí" o "no"
    if user_input_clean == "si":
        return "¿En qué puedo ayudarte?", "respuesta_si", 1.0
    elif user_input_clean == "no":
        return "Espero verte pronto.", "respuesta_no", 1.0

    # Procesar la entrada del usuario como de costumbre
    if any(palabra in user_input_clean for palabra in palabras_bien) and not any(neg_word in user_input_clean for neg_word in palabras_mal):
        return random.choice(respuestas_bien), "bien", 1.0
    elif any(neg_word in user_input_clean for neg_word in palabras_mal):
        return random.choice(respuestas_mal), "mal", 1.0

    top_intents, top_scores = await intent_detection(user_input)  # Await intent_detection function
    for intent, score in zip(top_intents, top_scores):
        print(f"Detected intent: {intent} with score: {score}")

    threshold = 0.9
    if top_scores[0] < threshold:
        return 'No entiendo. ¿Puedes reformular la pregunta?', top_intents[0], top_scores[0]
    for intent_obj in intents['intents']:
        if intent_obj['tag'] == top_intents[0]:
            responses = intent_obj['responses']
            return random.choice(responses), top_intents[0], top_scores[0]
    return 'No entiendo. ¿Puedes reformular la pregunta?', top_intents[0], top_scores[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    print("Ruta /chat llamada")
    user_input = request.json.get('message')
    print("Solicitud JSON:", request.json)
    response, intent, score = asyncio.run(respond_to_user(user_input, intents))  # Use asyncio.run() to run the function asynchronously
    print("Respuesta:", response)
    print("Intent:", intent)
    print("Score:", score)
    return jsonify({"response": str(response), "intent": str(intent), "score": float(score)})

if __name__ == '__main__':
    app.run(debug=True)
