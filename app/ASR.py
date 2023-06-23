# Description: contains the class used to Automatic Speech Recognition using finetuend-Whisper model
# Author: Ahmed Magdi and Ahmed Magdy
# Last Modified: 23-06-2023
# ------------------------------------------------------------------------------
from onnxruntime import InferenceSession
from transformers import AutoProcessor
import numpy as np
from .constants import DUMMY_FEED, PAST, EOS_TOKEN_ID, TIMEOUT
import time

# Color text in terminal


def color(text, color="default"):
    colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "default": 37}
    if color in colors.keys():
        return f"\033[{colors[color]}m{text}\033[0m"
    else:
        return text

# Print values with colors reresning the delay


def print_perf(key, value, great=1, good=3):
    if value < great:
        key = color(f"{key} : ", "blue")
        print(key + color(str(value), 'green'))
    elif value < good:
        key = color(f"{key} : ", "blue")
        print(key + color(str(value), 'yellow'))
    else:
        key = color(f"{key} : ", "blue")
        print(key + color(str(value), 'red'))


class Whisper:
    """
    A class used to Automatic Speech Recognition using finetuend-Whisper model

    ...

    Attributes
    ----------
    encoder_session : InferenceSession
        The encoder ort session
    decoder : InferenceSession
        The decoder ort session
    feature_extractor : WhisperFeatureExtractor
        The feature extractor
    tokenizer : WhisperTokenizer
        The number of legs the animal has (default 4)
    INIT_INPUT_IDS : np.array
        The initial input ids

    Methods
    -------
    load_models(encoder_model_path, decoder_model_path, model_name)
        Load the encoder ,decoder models, feature extractor and tokenizer
    extract_features(audio)
        Extract features from the audio
        return np.array
    encode(input_features)
        Encode the input features
        return np.array (encoder_hidden_states)
    decode_step(encoder_hidden_states, from_encoder=False, prev_decoder=None, last_token=50362)
        Decode a single step returning single token
        return np.array (decoder_output)(logits, past_key_values, encoder_hidden_states)
    get_token_id(decoder_out)
        Get the token id from decoder output
        return the index of the token with has the highest probability
    transcribe(audio_data)
        combine all functions and transcribe the audio to text using iterative decoding
        return text, perf(dict)
    process(audio_data=None, keywords=[], log=False)
        Combine all functions and decide whether student is cheating or not
        return True, text, perf(dict)
    """

    encoder_session = None
    decoder_session = None
    feature_extractor = None
    tokenizer = None
    INIT_INPUT_IDS = None

    @staticmethod
    def load_models(encoder_model_path, decoder_model_path, model_name):
        Whisper.encoder_session = InferenceSession(encoder_model_path)
        Whisper.decoder_session = InferenceSession(decoder_model_path)
        processor = AutoProcessor.from_pretrained(model_name)
        Whisper.feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer
        init_input_ids = tokenizer.get_decoder_prompt_ids()
        Whisper.tokenizer = tokenizer
        Whisper.INIT_INPUT_IDS = np.array(init_input_ids, dtype=np.int64)

    # Extract features from the audio
    def extract_features(self, audio):
        return np.array(Whisper.feature_extractor(audio, sampling_rate=16000).input_features, dtype=np.float32)

    # Encode the input features
    def encode(self, input_features):
        return Whisper.encoder_session.run(None, input_feed={"input_features": input_features})[0]

    # Decode a single step returning single token
    def decode_step(self, encoder_hidden_states, from_encoder=False, prev_decoder=None, last_token=50362):
        if from_encoder:
            input_feed = DUMMY_FEED
            input_feed["input_ids"] = Whisper.INIT_INPUT_IDS
            input_feed["encoder_hidden_states"] = encoder_hidden_states
            input_feed["use_cache_branch"] = np.array([False])

        else:
            input_feed = {PAST[i]: prev_decoder[i + 1]
                          for i in range(len(PAST))}
            input_feed["input_ids"] = np.array([last_token], dtype=np.int64)
            input_feed["encoder_hidden_states"] = encoder_hidden_states
            input_feed["use_cache_branch"] = np.array([False])

        return Whisper.decoder_session.run(None, input_feed=input_feed)

    # Get the token id from decoder output
    def get_token_id(self, decoder_out):
        logits = decoder_out[0]
        return np.argmax(logits, axis=-1).flatten()

    def is_keyword_present(self, keywords, text):
        result = []
        for keyword in keywords:
            if keyword in text:
                result.append(keyword)

        return result

    # Transcribe audio
    def transcribe(self, audio_data):
        perf = {}
        timeout = False
        total_time = time.time()
        start_time = time.time()

        # Step 1 | Feature extraction
        input_features = self.extract_features(audio_data)
        perf["feature_extraction"] = time.time() - start_time
        start_time = time.time()

        # Step 2 | Encoding input features
        encoder_hidden_states = self.encode(input_features)
        perf["encoding"] = time.time() - start_time
        start_time = time.time()

        # Step 3 | Decoding first step
        decoder_step = self.decode_step(
            encoder_hidden_states, from_encoder=True)
        token_id = self.get_token_id(decoder_step)
        tokens_ids = [token for token in token_id]

        # Step 4 | Perform iterative decoding to get the rest of text
        while not tokens_ids[-1] == EOS_TOKEN_ID:
            decoder_step = self.decode_step(encoder_hidden_states,
                                            prev_decoder=decoder_step, last_token=tokens_ids)
            token_id = self.get_token_id(decoder_step)
            tokens_ids.append(token_id[-1])

            if time.time() - start_time >= TIMEOUT:
                timeout = False
                break

        perf["decoding"] = time.time() - start_time
        text = Whisper.tokenizer.decode(tokens_ids, skip_special_tokens=True)

        if timeout:
            text += "...(TIMEOUT)"

        perf["timeout"] = timeout
        perf["total_time"] = time.time() - total_time

        return text, perf

    # Combine all functions and decide whether student is cheating or not

    def process(self, audio_data=None, keywords=[], log=False):
        text, perf = self.transcribe(audio_data)
        if log:
            print(f"{color('Transcroption :', 'blue')} {text}")
            for key in perf:
                if not key == "timeout":
                    print_perf(key, round(perf[key], 3))

            timeout = color("True", "red") if perf['timeout'] else color(
                "True", "green")
            print(f"timeout = {timeout}")

        used_keywords = self.is_keyword_present(keywords, text)
        isCheating = len(used_keywords) != 0

        for keyword in used_keywords:
            text = text.replace(keyword, f"[{keyword}]")

        return isCheating, text, perf
