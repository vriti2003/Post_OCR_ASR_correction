# Importing Libraries
import pandas as pd
import numpy as np
import os
from jiwer import *
import keras
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Layer, Reshape,GRU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.activations import softmax

# Selecting the GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Setting the seed for reproducibilty
tf.random.set_seed(45)
np.random.seed(45)

# Setting the hyperparametres
TIME_STEPS = 20
MAX_LEN_ORD = 4
LENGTH_PER_ORD = 10

# Setting the property to infinte print
np.set_printoptions(threshold = np.inf)

# Reading the train, val and test excel files
train_data=pd.read_excel("../hindi_data/new_data/new_hindi_train.xlsx")
test_data=pd.read_excel("../hindi_data/new_data/new_hindi_test.xlsx")
val_data=pd.read_excel("../hindi_data/new_data/new_hindi_val.xlsx")

# Reading the columns of imported excel files
train_ocr=list(train_data["ocr_output"])
train_c=list(train_data["correct_text"])
train_asr=list(train_data["asr_output"])

val_ocr=list(val_data["ocr_output"])
val_c=list(val_data["correct_text"])
val_asr=list(val_data["asr_output"])

test_ocr=list(test_data["ocr_output"])
test_c=list(test_data["correct_text"])
test_asr=list(test_data["asr_output"])


def unique_chars(words):
  """
    Returns unique chars for the given list of words to dictionary of frequency of unique characters.
    Args:
        words (list of strings): list of hindi words
    Returns:
        dictionary : return a dictionary of form : ('char' : 'frequency')
  """
  seen_char=[]
  for i in str(words):
    for j in str(i):
      if j not in seen_char:
        seen_char.append(j)
  return seen_char

def max_ord_len(unique_charsss):
  """
    Takes list of unique characters and returns maximum length of ord of characters
    Args:
        unique_charsss (list of strings): list of words
    Returns:
        int : return maximum length of ord
  """
  max_len=0
  for i in unique_charsss:
    if len(str(ord(i)))>max_len:
      max_len=len(str(ord(i)))
  return max_len

def cal_ord_words_padded(words, max_ord_length):

  """
    Calculate the ord of the words with padding
    Args:
        words (list of strings): list of hindi words
        max_ord_length (int): maximum length of ord of unique characters
    Returns:
        list : return a list of ord of words 
  """
  final_list=[]
  for word in words:
    o=[]
    for i in str(word):
      ord_i=str(ord(i))
      if len(ord_i)<max_ord_length:
        for j in range(max_ord_length-len(ord_i)):
          ord_i="0"+ord_i
      o.append(ord_i)
    if(len(str(word))<TIME_STEPS):
      for k in range(TIME_STEPS-len(str(word))):
        o.append("0"*max_ord_length)
    final_list.append(o)
  return final_list

def encode(ord_train_words):
  """
    Calculate the ohe of ord of the words with padding
    Args:
        ord_train_words (list of strings): list of ord of hindi words
    Returns:
        list : return a list of ohe of ord of words 
  """
  wordss = []
  for ord_words in ord_train_words:
    words = []
    for ord in ord_words:
      word = []
      for i in ord:
        x=np.zeros((10,1))
        x[int(i)]=1
        word.append(x)
      words.append(word)
    wordss.append(words)
  wordss = np.array(wordss).squeeze()
  return wordss


def decode_final(predict):

  """
    Generate actual word from the ohe of ord of the wordss
    Args:
        predict (list of strings): list of ohe of ord of hindi words
    Returns:
        list : return a list of actual words 
  """

  final_e=np.argmax(predict, 2)
  final_predict=[]
  for i in final_e:
    ord_word=""
    for j in i:
      ord_word+=str(j)

    ord_words = []
    startIndex = 0
    while startIndex < len(ord_word):
      ord_words.append(ord_word[startIndex: startIndex + 4])
      startIndex += 4
    word=""
    for k in ord_words:
      if(int(k)==0):
        break
      else:
        word+=chr(int(k))
    final_predict.append(word)

  return final_predict

def calculate_wer(train_predictions, train_targets):
    train_predictions_final=decode_final(train_predictions)
    train_targets_final=decode_final(train_targets)
    # print(train_predictions_final)
    return wer(train_targets_final, train_predictions_final)

def calculate_cer(train_predictions, train_targets):
    train_predictions_final=decode_final(train_predictions)
    train_targets_final=decode_final(train_targets)
    # print(train_targets_final)
    return cer(train_targets_final, train_predictions_final)

class HistorySaver(keras.callbacks.Callback):
    def __init__(self, train_data=None, val_data=None):
        super(HistorySaver, self).__init__()
        self.history = {}
        self.currentEpoch = 0
        self.train_data = train_data
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        # Save existing metrics in logs
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)

        # Compute WER and CER for training data
        train_predictions = self.model.predict(self.train_data[0])
        train_targets = self.train_data[1]
        # print(train_predictions)
        # print(train_targets)
        train_wer = calculate_wer(train_predictions, train_targets)
        train_cer = calculate_cer(train_predictions, train_targets)

        # Compute WER and CER for validation data
        val_predictions = self.model.predict(self.val_data[0])
        val_targets = self.val_data[1]

        val_wer = calculate_wer(val_predictions, val_targets)
        val_cer = calculate_cer(val_predictions, val_targets)

        # Add WER and CER to history
        self.history.setdefault('epoch', []).append(epoch)
        self.history.setdefault('train_wer', []).append(train_wer)
        self.history.setdefault('train_cer', []).append(train_cer)
        self.history.setdefault('val_wer', []).append(val_wer)
        self.history.setdefault('val_cer', []).append(val_cer)

        self.currentEpoch = epoch

    def on_train_end(self, logs=None):
        # Save the final state of history to the CSV
        file_path = 'track_cer_wer_train_val_after_epoch.csv'
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(file_path, mode='w', index=False)



"""### Train and Test"""
class Attention(Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Reshape, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Reshape, concatenate
from tensorflow.keras.models import Model

class AttentionGRU:
    def __init__(self, time_steps=20, max_ord_length=4, len_per_ord=10, gru_units=1024, dense_units=40, dropout=0.3, rec_dropout=0.3):
        self.time_steps = time_steps
        self.max_ord_length = max_ord_length
        self.len_per_ord = len_per_ord
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.rec_dropout = rec_dropout

    def build_model(self):
        input_ocr_output = Input(shape=(self.time_steps * self.max_ord_length, self.len_per_ord), name='input_ocr_output')
        input_asr_output = Input(shape=(self.time_steps * self.max_ord_length, self.len_per_ord), name='input_asr_output')

        # GRU layers for OCR and ASR outputs
        gru_ocr, hidden_ocr = GRU(self.gru_units, return_sequences=True, return_state=True, use_bias=True,
                                  recurrent_dropout=self.rec_dropout, name='gru_ocr')(input_ocr_output)
        gru_asr, hidden_asr = GRU(self.gru_units, return_sequences=True, return_state=True, use_bias=True,
                                  recurrent_dropout=self.rec_dropout, name='gru_asr')(input_asr_output)

        # Attention mechanism applied to the GRU outputs
        attention_layer = Attention(self.dense_units)
        context_vector_ocr, _ = attention_layer(gru_ocr, hidden_ocr)
        context_vector_asr, _ = attention_layer(gru_asr, hidden_asr)

        # Combine context vectors
        combined_context = concatenate([context_vector_ocr, context_vector_asr], axis=-1)
        dropout_layer = Dropout(self.dropout)(combined_context)

        # Final Dense layer to predict the output sequence
        output = Dense(self.time_steps * self.max_ord_length * self.len_per_ord, activation='softmax', name='output')(dropout_layer)
        # Reshape the output to match the target shape
        output = Reshape((self.time_steps * self.max_ord_length, self.len_per_ord))(output)

        # Compile model
        model = Model(inputs=[input_ocr_output, input_asr_output], outputs=[output])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
# calculate the unique chars
words = train_ocr + train_c + train_asr
uniq_words = unique_chars(words)

# calculate the max ord length of the char
max_length_ord=max_ord_len(uniq_words)

# calculate the ohe of ord of the words of train_ocr
# list_train_ocr_ord=cal_ord_words(train_ocr,max_length_ord)
list_train_ocr_ord=cal_ord_words_padded(train_ocr,max_length_ord)
list_train_ocr_ord_ohe=encode(list_train_ocr_ord)

# calculate the ohe of ord of the words of train_asr
# list_train_asr_ord=cal_ord_words(train_asr,max_length_ord)
list_train_asr_ord=cal_ord_words_padded(train_asr,max_length_ord)
list_train_asr_ord_ohe=encode(list_train_asr_ord)

# calculate the ohe of ord of the words of train_correct
# list_train_c_ord=cal_ord_words(train_c,max_length_ord)
list_train_c_ord=cal_ord_words_padded(train_c,max_length_ord)
list_train_c_ord_ohe=encode(list_train_c_ord)

list_train_ocr_ord_ohe_reshaped=list_train_ocr_ord_ohe.reshape(list_train_ocr_ord_ohe.shape[0],list_train_ocr_ord_ohe.shape[1]*list_train_ocr_ord_ohe.shape[2],list_train_ocr_ord_ohe.shape[3])
list_train_asr_ord_ohe_reshaped=list_train_asr_ord_ohe.reshape(list_train_asr_ord_ohe.shape[0],list_train_asr_ord_ohe.shape[1]*list_train_asr_ord_ohe.shape[2],list_train_asr_ord_ohe.shape[3])
list_train_c_ord_ohe_reshaped=list_train_c_ord_ohe.reshape(list_train_c_ord_ohe.shape[0],list_train_c_ord_ohe.shape[1]*list_train_c_ord_ohe.shape[2],list_train_c_ord_ohe.shape[3])

# calculate the ohe of ord of the words of val_ocr
# list_val_ocr_ord=cal_ord_words(val_ocr,max_length_ord)
list_val_ocr_ord=cal_ord_words_padded(val_ocr,max_length_ord)
list_val_ocr_ord_ohe=encode(list_val_ocr_ord)

# calculate the ohe of ord of the words of val_asr
list_val_asr_ord=cal_ord_words_padded(val_asr,max_length_ord)
list_val_asr_ord_ohe=encode(list_val_asr_ord)

# calculate the ohe of ord of the words of val_correct
list_val_c_ord=cal_ord_words_padded(val_c,max_length_ord)
list_val_c_ord_ohe=encode(list_val_c_ord)

list_val_ocr_ord_ohe_reshaped=list_val_ocr_ord_ohe.reshape(list_val_ocr_ord_ohe.shape[0],list_val_ocr_ord_ohe.shape[1]*list_val_ocr_ord_ohe.shape[2],list_val_ocr_ord_ohe.shape[3])
list_val_asr_ord_ohe_reshaped=list_val_asr_ord_ohe.reshape(list_val_asr_ord_ohe.shape[0],list_val_asr_ord_ohe.shape[1]*list_val_asr_ord_ohe.shape[2],list_val_asr_ord_ohe.shape[3])
list_val_c_ord_ohe_reshaped=list_val_c_ord_ohe.reshape(list_val_c_ord_ohe.shape[0],list_val_c_ord_ohe.shape[1]*list_val_c_ord_ohe.shape[2],list_val_c_ord_ohe.shape[3])

# calculate the ohe of ord of the words of test_ocr
list_test_ocr_ord=cal_ord_words_padded(test_ocr,max_length_ord)
list_test_ocr_ord_ohe=encode(list_test_ocr_ord)

# calculate the ohe of ord of the words of test_asr
list_test_asr_ord=cal_ord_words_padded(test_asr,max_length_ord)
list_test_asr_ord_ohe=encode(list_test_asr_ord)

#reshaping input to make it model compatible
list_test_ocr_ord_ohe_reshaped=list_test_ocr_ord_ohe.reshape(list_test_ocr_ord_ohe.shape[0],list_test_ocr_ord_ohe.shape[1]*list_test_ocr_ord_ohe.shape[2],list_test_ocr_ord_ohe.shape[3])
list_test_asr_ord_ohe_reshaped=list_test_asr_ord_ohe.reshape(list_test_asr_ord_ohe.shape[0],list_test_asr_ord_ohe.shape[1]*list_test_asr_ord_ohe.shape[2],list_test_asr_ord_ohe.shape[3])

# Initialize and compile model

attention_GRU = AttentionGRU(time_steps=TIME_STEPS, max_ord_length=MAX_LEN_ORD, len_per_ord = LENGTH_PER_ORD, gru_units=1024, dense_units=64, dropout=0.5, rec_dropout=0.5)
model= attention_GRU.build_model()

train_data = ({"input_ocr_output": list_train_ocr_ord_ohe_reshaped, 'input_asr_output': list_train_asr_ord_ohe_reshaped},list_train_c_ord_ohe_reshaped)
val_data = ({"input_ocr_output": list_val_ocr_ord_ohe_reshaped, 'input_asr_output': list_val_asr_ord_ohe_reshaped}, list_val_c_ord_ohe_reshaped)

CHECKPOINT_PATH = 'checkpoint/'
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_PATH, 'model-{epoch:03d}.keras'), 
    save_weights_only=False, 
    verbose=1, 
    save_best_only=False, 
    save_freq='epoch' 
)

early_stopping_cp = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)

'''Training start'''
history = model.fit(
    [list_train_ocr_ord_ohe_reshaped, list_train_asr_ord_ohe_reshaped],
    list_train_c_ord_ohe_reshaped,
    validation_data=([list_val_ocr_ord_ohe_reshaped, list_val_asr_ord_ohe_reshaped], list_val_c_ord_ohe_reshaped),
    epochs=200,
    batch_size=512,
    callbacks=[cp_callback, HistorySaver(train_data, val_data), early_stopping_cp])

'''Saving the Loss curve plot'''
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('train_val_loss_curve.png')
