import numpy as np

# Configurations for whisper-base.en, can be found in model/config.json
BATCH_SIZE = 1
DECODER_HEADS = 8
ENCODER_HEADS = 8
D_MODEL = 512
DECODER_SEQUENCE_SIZE = 2
ENCODER_SEQUENCE_SIZE = 1500
DECODER_D_MODEL_PER_HEADS = int(D_MODEL / DECODER_HEADS)
ENCODER_D_MODEL_PER_HEADS = int(D_MODEL / ENCODER_HEADS)
EOS_TOKEN_ID = 50256

DECODER_DIM = [BATCH_SIZE, DECODER_HEADS,
               DECODER_SEQUENCE_SIZE, DECODER_D_MODEL_PER_HEADS]
ENCODER_DIM = [BATCH_SIZE, ENCODER_HEADS,
               ENCODER_SEQUENCE_SIZE, ENCODER_D_MODEL_PER_HEADS]

PAST = ["past_key_values.0.decoder.key",
        "past_key_values.0.decoder.value",
        "past_key_values.0.encoder.key",
        "past_key_values.0.encoder.value",
        "past_key_values.1.decoder.key",
        "past_key_values.1.decoder.value",
        "past_key_values.1.encoder.key",
        "past_key_values.1.encoder.value",
        "past_key_values.2.decoder.key",
        "past_key_values.2.decoder.value",
        "past_key_values.2.encoder.key",
        "past_key_values.2.encoder.value",
        "past_key_values.3.decoder.key",
        "past_key_values.3.decoder.value",
        "past_key_values.3.encoder.key",
        "past_key_values.3.encoder.value",
        "past_key_values.4.decoder.key",
        "past_key_values.4.decoder.value",
        "past_key_values.4.encoder.key",
        "past_key_values.4.encoder.value",
        "past_key_values.5.decoder.key",
        "past_key_values.5.decoder.value",
        "past_key_values.5.encoder.key",
        "past_key_values.5.encoder.value"
        ]

PRESENT = [value.replace("past_key_values", "present") for value in PAST]

DUMMY_FEED = {}
for key in PAST:
    if "decoder" in key:
        DUMMY_FEED[key] = np.empty(DECODER_DIM, dtype=np.float32)
    else:
        DUMMY_FEED[key] = np.empty(ENCODER_DIM, dtype=np.float32)


TIMEOUT = 15  # 15 seconds
