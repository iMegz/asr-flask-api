# Whisper Automatic Speech Recognition API Using Flask
 
This documentation provides information on how to interact with the server using Python and React JS.

## Introduction

This server is developed by [A. Magdi Mostafa](https://github.com/iMegz) and [A. Magdy Fahmy](https://github.com/AMF777). It utilizes a fine-tuned [Whisper-base.en](https://huggingface.co/AhmedMEGZ/whisper-finetuned/tree/main) model for speech recognition.

## Connecting to the Server

To connect to the server, you can use the following code snippets in Python and React JS.

### Python

```python
import socketio

sio = socketio.Client()
sio.connect('http://localhost:5000')

@sio.on('connected')
def on_connected(data):
    print(data)

@sio.on('isCheating')
def on_cheating_detected(text):
    print(f'Cheating detected: {text}')

# Send data to the server
data = {
    'audio_blob': audio_blob,
    'keywords': keywords
}
sio.emit('data', data)
```

### ReactJS

```js
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

socket.on('connected', (data) => {
    console.log(data);
});

socket.on('isCheating', (text) => {
    console.log('Cheating detected:', text);
});

// Send data to the server
const data = {
    audio_blob: audioBlob,
    keywords: keywords
};
socket.emit('data', data);
```

### Data types

The server expects the following data types:

* `audio_blob`: A `Float32Array` buffer containing the audio data in WAV format.
 *` keywords`: An array of strings representing the keywords.

### Running the Server

To run the server, navigate to the project directory in the terminal and execute the following command:
```bash
python app.py
```
