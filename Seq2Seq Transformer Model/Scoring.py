# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:07:18 2021

@author: kesha
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import preprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import pickle
import os
os.chdir("C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Instacart/Code/Seq2Seq")
from Transformer import create_model

run_type = "score"
from_scratch = True
wd = "C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Instacart/Run_Seq2Seq/"
raw_data_loc = "C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Instacart/"

tmp = pd.read_csv(wd + "score_universe.csv", converters={'user_id': str, 'NEXT_TXN_DAYS': str, 'NEXT_TXN_DOW': str, 'NEXT_TXN_HOD': str}, chunksize=1000, iterator=True) # nrows=100000
scoring_universe = pd.concat(tmp, ignore_index=True)
del tmp
gc.collect()

config = pickle.load(open(wd+"config.pkl", "rb"))
pickle_byte_obj = pickle.load(open(wd+"tokenizer.pkl", "rb"))
tokenizer_items = pickle_byte_obj['items']
tokenizer_cids = pickle_byte_obj['cids']
tokenizer_days = pickle_byte_obj['days']
tokenizer_aisles = pickle_byte_obj['aisles']
tokenizer_department = pickle_byte_obj['department']
tokenizer_dow = pickle_byte_obj['dow']
tokenizer_hod = pickle_byte_obj['hod']


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
gpus = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(gpus))
# Open a strategy scope.
if gpus > 1:
    with strategy.scope():
        recommendation_model = create_model(config)
        recommendation_model.load_weights(wd + "keras-recommender.h5")
else:
    recommendation_model = create_model(config)
    recommendation_model.load_weights(wd + "keras-recommender.h5")
    
    
def get_inference(model_universe, config, index):
    # Inference
    # target = model_universe.loc[index, ['NEXT_TXN']].tolist()
    # Items
    sentence = model_universe.loc[index, ['TXN_HISTORY']].tolist()
    txt_to_seq = tokenizer_items.texts_to_sequences(sentence) 
    sentence_to_list = txt_to_seq[0]
    encoder_items = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
    # Days
    txt_to_seq = tokenizer_days.texts_to_sequences(model_universe.loc[index, ['TXN_HISTORY_DAYS']].tolist()) 
    encoder_txn_days = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
    # Customer ID
    txt_to_seq = tokenizer_cids.texts_to_sequences(model_universe.loc[index, ['user_id']].tolist()) 
    encoder_cids = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
    # Next Days
    txt_to_seq = tokenizer_days.texts_to_sequences(model_universe.loc[index, ['NEXT_TXN_DAYS']].tolist()) 
    encoder_nxt_txn_days = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
    # Aisles
    txt_to_seq = tokenizer_aisles.texts_to_sequences(model_universe.loc[index, ['TXN_HISTORY_AISLE']].tolist()) 
    encoder_txn_aisle = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
    # Dept
    txt_to_seq = tokenizer_department.texts_to_sequences(model_universe.loc[index, ['TXN_HISTORY_DEPT']].tolist()) 
    encoder_txn_dept = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
    # DOW
    txt_to_seq = tokenizer_dow.texts_to_sequences(model_universe.loc[index, ['TXN_HISTORY_DOW']].tolist()) 
    encoder_txn_dow = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
    # HOD
    txt_to_seq = tokenizer_hod.texts_to_sequences(model_universe.loc[index, ['TXN_HISTORY_HOD']].tolist()) 
    encoder_txn_hod = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
    # Next DOW
    txt_to_seq = tokenizer_dow.texts_to_sequences(model_universe.loc[index, ['NEXT_TXN_DOW']].tolist()) 
    encoder_nxt_txn_dow = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
    # HOD
    txt_to_seq = tokenizer_hod.texts_to_sequences(model_universe.loc[index, ['NEXT_TXN_HOD']].tolist()) 
    encoder_nxt_txn_hod = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
    
    translation = ['sot']
    predicted_sentence = []
    for i in range(min(len(sentence_to_list), 100)):
        if i == 0:
            txt_to_seq = tokenizer_items.texts_to_sequences(['sot'])
        else:
            txt_to_seq = tokenizer_items.texts_to_sequences(translation)
        decoder_items = pad_sequences(txt_to_seq, maxlen=len(txt_to_seq[0]), padding='pre')
        predict = recommendation_model.predict({'input_1': encoder_items, 'input_2': decoder_items, 'input_3': encoder_txn_days, 'input_4': encoder_cids, 'input_5': encoder_nxt_txn_days, 'input_6': encoder_txn_aisle, 'input_7': encoder_txn_dept, 'input_8': encoder_txn_dow, 'input_9': encoder_txn_hod, 'input_10': encoder_nxt_txn_dow, 'input_11': encoder_nxt_txn_hod})
        predicted_item = tokenizer_items.index_word[np.argmax(predict[-1,-1])]
        # predict[-1, -1].argsort()[-3:][::-1] # Top K
        if i == 0 and predicted_item == 'None':
            predicted_sentence.append(predicted_item)
            translation = [translation[0] + ' ' + predicted_item]
            break
        if predicted_item == 'eot':
            break    
        if predicted_item not in sentence[0] or predicted_item in predicted_sentence:
            for x in range(2, predict.shape[2]):
                predicted_item = tokenizer_items.index_word[predict[-1, -1].argsort()[-x]]
                if predicted_item == 'eot' or predicted_item == 'None':
                    break
                if predicted_item in sentence[0] and predicted_item not in predicted_sentence:
                    predicted_sentence.append(predicted_item)
                    translation = [translation[0] + ' ' + predicted_item]
                    break
        else:
            predicted_sentence.append(predicted_item)
            translation = [translation[0] + ' ' + predicted_item]
    
    # print("Input: ", sentence)
    # print("Ground Truth: ", target)
    # print("Predicted: ", translation)
    predicted_sentence = ' '.join(predicted_sentence)
    # print("Predicted: ", predicted_sentence)
    
    return predicted_sentence


# predicted_sentence = get_inference(scoring_universe, config, index=5)


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

predictions = []
for i in range(len(scoring_universe)):
    predicted_sentence = get_inference(scoring_universe, config, index=i)
    predictions.append(predicted_sentence) 
    printProgressBar(i + 1, total = len(scoring_universe), prefix = 'Progress:', suffix = 'Complete', decimals=4, length = 50)

scoring_universe['products'] = predictions
submission = scoring_universe[["order_id", "products"]]
submission.to_csv(wd+"submission.csv", index=False, header=True)