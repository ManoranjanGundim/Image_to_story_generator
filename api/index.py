import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import pickle
from transformers import pipeline, set_seed
import os

app = Flask(__name__, template_folder='../templates', static_folder='../static')

MAX_LENGTH = 49
ATTENTION_FEATURES_SHAPE = 64
EMBEDDING_DIM = 256
UNITS = 512
VOCAB_SIZE = 5001

image_features_extract_model = None
encoder = None
decoder = None
tokenizer = None
story_generator = None

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
    attention_weights = tf.nn.softmax(self.V(score), axis=1)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    context_vector, attention_weights = self.attention(features, hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

def load_models():
    global image_features_extract_model, encoder, decoder, tokenizer, story_generator
    
    if image_features_extract_model is None:
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    if tokenizer is None:
        with open('./models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

    if encoder is None:
        encoder = CNN_Encoder(EMBEDDING_DIM)
        encoder.load_weights('./models/encoder_weights.h5')

    if decoder is None:
        decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, VOCAB_SIZE)
        decoder.load_weights('./models/decoder_weights.h5')
    
    if story_generator is None:
        story_generator = pipeline('text-generation', model='gpt2')
        set_seed(42)

def load_image(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def generate_caption(image_bytes):
    load_models()
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image_bytes), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(MAX_LENGTH):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        word = tokenizer.index_word.get(predicted_id, '')
        if word == '<end>':
            break
        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)
    
    caption = ' '.join(result)
    return caption.capitalize()

def generate_story(caption):
    load_models()
    prompt = f"Once upon a time, {caption.lower()}."
    story = story_generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
    
    last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
    if last_punc != -1:
        story = story[:last_punc+1]

    return story

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        caption = generate_caption(img_bytes)
        story = generate_story(caption)
        return jsonify({'caption': caption, 'story': story})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)