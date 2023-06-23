# Description: This file contains the server-side code for the web application.
# Author: Ahmed Magdi and Ahmed Magdy
# Last Modified: 23-06-2023
# ------------------------------------------------------------------------------

# import the necessary packages
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import numpy as np
from ASR import Whisper
# from eval import calculate_wer # For testing purposes

app = Flask(__name__)
app.config['SECRET_KEY'] = 'megzers4ever'
socketio = SocketIO(app, cors_allowed_origins="*")

models_dir = "models/"
enocder_path = models_dir + "encoder.ort"
decoder_path = models_dir + "decoder.ort"

model_name = "AhmedMEGZ/whisper-finetuned"

log_to_client = True


@app.route('/')
def index():
    """
    Renders the index.html template.
    """
    print("Server on")
    return render_template('index.html')


@socketio.on('connect')
def connected():
    """
    Renders the index.html template.
    """
    socketio.emit("connected", "Connection confirmed")


@socketio.on('data')
def handle_data(data):
    """
    Handles the 'data' event.

    Parameters:
        data (dict): The data received from the client containing the audio blob and keywords.

    Data Types:
        - data['audio_blob'] (bytes): A float32 buffer of a WAV file.
        - data['keywords'] (list of str): An array of keywords.
    """
    audio_blob = data['audio_blob']
    keywords = data['keywords']
    model = Whisper()
    audio_data = np.frombuffer(audio_blob, dtype=np.float32)

    # Perform speech recognition using the loaded models and sessions
    isCheating, text, perf = model.process(audio_data, keywords)

    if log_to_client:
        socketio.emit('log', {'text': text, 'perf': perf})

    if isCheating:
        # Emit a success message back to the frontend
        socketio.emit('isCheating', text)





if __name__ == "__main__":
    print('Server starting')
    from time import time as t
    start = t()
    Whisper.load_models(enocder_path, decoder_path, model_name)
    print(f'Models loaded in {round(t() - start, 3)}s')
    print('Server starting')
    socketio.run(app, host="0.0.0.0")