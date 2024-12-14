# importing libraries
import pandas as pd
import numpy as np
import os
from jiwer import *
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Layer, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

#This attention class is same that was used during training
class Attention(Layer):
  def __init__(self,units,**kwargs):
    super(Attention,self).__init__(**kwargs)
    self.W1=Dense(units)
    self.W2=Dense(units)
    self.V=Dense(1)

  def call(self,features,hidden):
    hidden_with_time_axis=tf.expand_dims(hidden,1)
    score = tf.nn.tanh(self.W1(features)+self.W2(hidden_with_time_axis))
    attention_weights=tf.nn.softmax(self.V(score),axis=1)
    context_vector=attention_weights*features
    context_vector=tf.reduce_sum(context_vector,axis=1)
    return context_vector, attention_weights
  
# HYPERPARAMETRES
TIME_STEPS = 20
MAX_LEN_ORD = 4
LENGTH_PER_ORD = 10

test_data=pd.read_excel("../hindi_test.xlsx")
parent_dir = '26_epoch'
os.makedirs(parent_dir)

test_ocr=list(test_data["ocr_output"])
test_c=list(test_data["correct_text"])
test_asr=list(test_data["asr_output"])

"""#ENCODING"""

def unique_chars(words):
  """
    Returns unique chars for the given list of words to dictionary of frequency of unique characters.
    Args:
        words (list of strings): list of hindi words
    Returns:
        dictionary : return a dictionary of form : ('char' : 'frequency')
  """

  seen_char=[]
  for i in words:
    for j in i:
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
    for i in word:
      ord_i=str(ord(i))
      if len(ord_i)<max_ord_length:
        for j in range(max_ord_length-len(ord_i)):
          ord_i="0"+ord_i
      o.append(ord_i)
    if(len(word)<TIME_STEPS):
      for k in range(TIME_STEPS-len(word)):
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
  
max_length_ord=4

# calculate the ohe of ord of the words of test_ocr
list_test_ocr_ord=cal_ord_words_padded(test_ocr,max_length_ord)
list_test_ocr_ord_ohe=encode(list_test_ocr_ord)

# calculate the ohe of ord of the words of test_asr
list_test_asr_ord=cal_ord_words_padded(test_asr,max_length_ord)
list_test_asr_ord_ohe=encode(list_test_asr_ord)

list_test_ocr_ord_ohe_reshaped=list_test_ocr_ord_ohe.reshape(list_test_ocr_ord_ohe.shape[0],list_test_ocr_ord_ohe.shape[1]*list_test_ocr_ord_ohe.shape[2],list_test_ocr_ord_ohe.shape[3])
list_test_asr_ord_ohe_reshaped=list_test_asr_ord_ohe.reshape(list_test_asr_ord_ohe.shape[0],list_test_asr_ord_ohe.shape[1]*list_test_asr_ord_ohe.shape[2],list_test_asr_ord_ohe.shape[3])

#Define the custom objects mapping
custom_objects={'Attention':Attention}
path='checkpoint/epoch_07.keras'

#Load the model using the custom objects
model=load_model(path,custom_objects=custom_objects)

# set model to eval mode
model.trainable=False

# disable training for all layers explicitly
for layer in model.layers:
  layer.trainable=False

#clear the optimizer
model.optimizer=None

predict=model.predict([list_test_ocr_ord_ohe_reshaped, list_test_asr_ord_ohe_reshaped])
decode_predictions = decode_final(predict)

correct_samples_cnt = 0
incorrect_samples_cnt = 0

gt_samples = []
incorrect_samples = []
cer_=[]
wer_=[]
hyp_ = []
ref_ = []

output_seq = decode_predictions
original_output_seq = test_c

with open(os.path.join(parent_dir, 'qualitative_analysis_model.txt',"w")) as q:

    for i in range(len(output_seq)):
      hyp_seq = output_seq[i] # predict seq
      ref_seq = original_output_seq[i] # reference/original seq
      q.write(f"Generated: {hyp_seq}, Reference: {ref_seq}")
      hyp_.append(hyp_seq)
      ref_.append(ref_seq)
      cer_cal = cer(ref_seq, hyp_seq)
      wer_cal = wer(ref_seq, hyp_seq)
      cer_.append(cer_cal)
      wer_.append(wer_cal)
      if (wer_cal == 0.0): 
        correct_samples_cnt += 1
      else:
        incorrect_samples_cnt += 1
      incorrect_samples.append(hyp_seq)
      gt_samples.append(ref_seq)
      q.write(f"CER: {cer_cal}, WER: {wer_cal}")
      q.write("=" * 50)

with open(os.path.join(parent_dir, "model_output_model.txt", "w")) as op:
    op.write("\n")
    op.write("Count of correct samples, incorrect samples, total samples  ")
    op.write(f"{correct_samples_cnt}, {incorrect_samples_cnt}, {correct_samples_cnt + incorrect_samples_cnt}")
    op.write("\n")
    op.write(f"Percentage of correct samples: {(correct_samples_cnt/(correct_samples_cnt + incorrect_samples_cnt)) * 100}")
    op.write("\n")
    op.write(f"len(CER) : {len(cer_)} len(WER) : {len(wer_)}")
    op.write(f"Average Character Error Rate: {sum(cer_)/len(cer_)}")
    op.write(f"Average Word Error Rate:  {sum(wer_)/len(wer_)}")

with open(os.path.join(parent_dir, "output_model.csv", "w")) as f:
    f.write("Original target , Predicted target ,Prediction")
    f.write("\n")
    for i, (org, pred) in enumerate(zip(ref_, hyp_)):
        wer_cal = wer(pred, org)
        if (wer_cal == 0.0):
            #print(org, pred, "Correct")
            f.write(f"{org}, {pred},  Correct")
            f.write("\n")
        else:
            f.write(f"{org}, {pred},  Incorrect")
            f.write("\n")

