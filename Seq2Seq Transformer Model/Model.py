# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:58:04 2021

@author: kesha

! pip install pandas
! pip install numpy
! pip install sklearn
! pip install fsspec
! pip install s3fs
! pip install boto3

# Read pickle files from s3
import pickle
import boto3
s3 = boto3.resource('s3')
pickle_byte_obj = pickle.loads(s3.Bucket("keshav-datasets").Object("Instacart/tokenizer.pkl").get()['Body'].read())

# Write pickle file to s3
pickle_byte_obj = pickle.dumps(config)
s3_resource = boto3.resource('s3')
s3_resource.Object("keshav-datasets","Instacart/config.pkl").put(Body=pickle_byte_obj)
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

run_type = "model"
from_scratch = True
wd = "C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Instacart/Run_Seq2Seq/"
raw_data_loc = "C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Instacart/"

tmp = pd.read_csv(wd + "model_universe.csv", converters={'user_id': str, 'NEXT_TXN_DAYS': str, 'NEXT_TXN_DOW': str, 'NEXT_TXN_HOD': str}, nrows=10000, chunksize=1000, iterator=True) # nrows=100000
model_universe = pd.concat(tmp, ignore_index=True)
del tmp
gc.collect()

pickle_byte_obj = pickle.load(open(wd+"tokenizer.pkl", "rb"))
tokenizer_items = pickle_byte_obj['items']
tokenizer_cids = pickle_byte_obj['cids']
tokenizer_days = pickle_byte_obj['days']
tokenizer_aisles = pickle_byte_obj['aisles']
tokenizer_department = pickle_byte_obj['department']
tokenizer_dow = pickle_byte_obj['dow']
tokenizer_hod = pickle_byte_obj['hod']

if from_scratch:
    config = {
        'MAX_ITEMS_PER_TXN': 100,
        'ENCODER_MAX_LEN': 300,
        'DECODER_MAX_LEN': 100,
        'BATCH_SIZE': 8, #32
        'EPOCHS': 2, # 12
        'LR': 0.001,
        'EMBED_DIM': 128,
        'NUM_HEAD': 8,  
        'FF_DIM': 768,
        'NUM_LAYERS': 2,
        'SAMPLES_PER_EPOCH':250000,
        'ITEM_VOCAB_SIZE': int(len(tokenizer_items.word_index)),
        'CID_VOCAB_SIZE': int(len(tokenizer_cids.word_index)),
        'DAYS_VOCAB_SIZE': int(len(tokenizer_days.word_index)),
        'AISLES_VOCAB_SIZE': int(len(tokenizer_aisles.word_index)),
        'DEPT_VOCAB_SIZE': int(len(tokenizer_department.word_index)),
        'DOW_VOCAB_SIZE': int(len(tokenizer_dow.word_index)),
        'HOD_VOCAB_SIZE': int(len(tokenizer_hod.word_index))
        }
    
    pickle_byte_obj = [config]
    pickle.dump(pickle_byte_obj, open(wd + "config.pkl", "wb"))
else:
    config = pickle.load(open(wd+"config.pkl", "rb"))[0]


class DataGenerator(keras.utils.Sequence):
    'Batch Generator for Keras'
    def __init__(self, universe, batch_size=32, runType="model", shuffle=True):   
        'Initialization'
        self.universe = universe
        self.model_universe = None
        self.nrows = len(self.universe) if config["SAMPLES_PER_EPOCH"] > len(self.universe) else config["SAMPLES_PER_EPOCH"]
        self.batch_size = batch_size
        self.run_type = runType
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nrows / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch = self.model_universe[index*self.batch_size:(index+1)*self.batch_size].index
        
        if self.run_type.lower() == "model":
            encoder_items, decoder_items, encoder_txn_days, encoder_cids, encoder_nxt_txn_days, encoder_txn_aisle, encoder_txn_dept, encoder_txn_dow, encoder_txn_hod, encoder_nxt_txn_dow, encoder_nxt_txn_hod = self.data_preprocess(batch)
            X = {"input_1": encoder_items, 'input_2': decoder_items[:, :-1], 'input_3': encoder_txn_days, 'input_4': encoder_cids, 'input_5': encoder_nxt_txn_days, 'input_6': encoder_txn_aisle, 'input_7': encoder_txn_dept, 'input_8': encoder_txn_dow, 'input_9': encoder_txn_hod, 'input_10': encoder_nxt_txn_dow, 'input_11': encoder_nxt_txn_hod}
            Y = {'output_1': decoder_items[:, 1:]}
            return X,Y

    def on_epoch_end(self):
        if self.shuffle:
            self.universe = self.universe.sample(frac=1).reset_index(drop=True)
        self.model_universe = resample(self.universe, replace=False, n_samples=self.nrows).reset_index(drop=True)
        print("Data Generator: Epoch End")
            
    def data_preprocess(self, batch):
        'Processes data in batch_size samples'
        
        # Items
        universe_sample = self.model_universe.iloc[batch]
        txt_to_seq = tokenizer_items.texts_to_sequences(universe_sample['TXN_HISTORY'].tolist()) 
        encoder_items = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
        txt_to_seq = tokenizer_items.texts_to_sequences(universe_sample['NEXT_TXN'].tolist()) 
        decoder_items = pad_sequences(txt_to_seq, maxlen=config['DECODER_MAX_LEN'], padding='pre')
        # Days
        txt_to_seq = tokenizer_days.texts_to_sequences(universe_sample['TXN_HISTORY_DAYS'].tolist()) 
        encoder_txn_days = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
        # Customer ID
        txt_to_seq = tokenizer_cids.texts_to_sequences(universe_sample['user_id'].tolist()) 
        encoder_cids = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
        # Next Days
        txt_to_seq = tokenizer_days.texts_to_sequences(universe_sample['NEXT_TXN_DAYS'].tolist()) 
        encoder_nxt_txn_days = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
        # Aisles
        txt_to_seq = tokenizer_aisles.texts_to_sequences(universe_sample['TXN_HISTORY_AISLE'].tolist()) 
        encoder_txn_aisle = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
        # Dept
        txt_to_seq = tokenizer_department.texts_to_sequences(universe_sample['TXN_HISTORY_DEPT'].tolist()) 
        encoder_txn_dept = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
        # DOW
        txt_to_seq = tokenizer_dow.texts_to_sequences(universe_sample['TXN_HISTORY_DOW'].tolist()) 
        encoder_txn_dow = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
        # HOD
        txt_to_seq = tokenizer_hod.texts_to_sequences(universe_sample['TXN_HISTORY_HOD'].tolist()) 
        encoder_txn_hod = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
        # Next DOW
        txt_to_seq = tokenizer_dow.texts_to_sequences(universe_sample['NEXT_TXN_DOW'].tolist()) 
        encoder_nxt_txn_dow = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
        # HOD
        txt_to_seq = tokenizer_hod.texts_to_sequences(universe_sample['NEXT_TXN_HOD'].tolist()) 
        encoder_nxt_txn_hod = pad_sequences(txt_to_seq, maxlen=1, padding='pre')

        return encoder_items, decoder_items, encoder_txn_days, encoder_cids, encoder_nxt_txn_days, encoder_txn_aisle, encoder_txn_dept, encoder_txn_dow, encoder_txn_hod, encoder_nxt_txn_dow, encoder_nxt_txn_hod

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
gpus = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(gpus))
# Open a strategy scope.
if gpus > 1:
    with strategy.scope():
        recommendation_model = create_model(config)
        if from_scratch == False:
            recommendation_model.load_weights(wd + "keras-recommender.h5")
else:
    recommendation_model = create_model(config)
    if from_scratch == False:
        recommendation_model.load_weights(wd + "keras-recommender.h5")
reduce_learning_rate = ReduceLROnPlateau(monitor = "val_sparse_categorical_accuracy", factor = 0.6, patience=1, min_lr=1e-5,  verbose=1, mode = 'max')
early_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', verbose=1, patience=3, restore_best_weights = True)
callbacks_list = [reduce_learning_rate, early_stopping]

# Train-test split
learn_universe, validation_universe = train_test_split(model_universe, test_size = 0.1, random_state = 0)
del model_universe
gc.collect()
learn_universe.reset_index(drop=True, inplace=True)
validation_universe.reset_index(drop=True, inplace=True)

learn_generator = DataGenerator(learn_universe, batch_size=config['BATCH_SIZE'], runType="model", shuffle=True)
validation_generator = DataGenerator(validation_universe, batch_size=config['BATCH_SIZE'], runType="model", shuffle=True)

recommendation_model.fit(learn_generator, validation_data = validation_generator, 
                              steps_per_epoch=int(learn_generator.nrows//config['BATCH_SIZE']), 
                              validation_steps = int(validation_generator.nrows//config['BATCH_SIZE']),
                              epochs=config['EPOCHS'], callbacks=[callbacks_list])

recommendation_model.save_weights(wd+"keras-recommender.h5")
# recommendation_model.save(wd+"keras-recommender-model.h5")





# Inference
index = 1
target = validation_universe.loc[index, ['NEXT_TXN']].tolist()
# Items
sentence = validation_universe.loc[index, ['TXN_HISTORY']].tolist()
txt_to_seq = tokenizer_items.texts_to_sequences(sentence) 
sentence_to_list = txt_to_seq[0]
encoder_items = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
# Days
txt_to_seq = tokenizer_days.texts_to_sequences(validation_universe.loc[index, ['TXN_HISTORY_DAYS']].tolist()) 
encoder_txn_days = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
# Customer ID
txt_to_seq = tokenizer_cids.texts_to_sequences(validation_universe.loc[index, ['user_id']].tolist()) 
encoder_cids = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
# Next Days
txt_to_seq = tokenizer_days.texts_to_sequences(validation_universe.loc[index, ['NEXT_TXN_DAYS']].tolist()) 
encoder_nxt_txn_days = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
# Aisles
txt_to_seq = tokenizer_aisles.texts_to_sequences(validation_universe.loc[index, ['TXN_HISTORY_AISLE']].tolist()) 
encoder_txn_aisle = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
# Dept
txt_to_seq = tokenizer_department.texts_to_sequences(validation_universe.loc[index, ['TXN_HISTORY_DEPT']].tolist()) 
encoder_txn_dept = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
# DOW
txt_to_seq = tokenizer_dow.texts_to_sequences(validation_universe.loc[index, ['TXN_HISTORY_DOW']].tolist()) 
encoder_txn_dow = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
# HOD
txt_to_seq = tokenizer_hod.texts_to_sequences(validation_universe.loc[index, ['TXN_HISTORY_HOD']].tolist()) 
encoder_txn_hod = pad_sequences(txt_to_seq, maxlen=config['ENCODER_MAX_LEN'], padding='pre')
# Next DOW
txt_to_seq = tokenizer_dow.texts_to_sequences(validation_universe.loc[index, ['NEXT_TXN_DOW']].tolist()) 
encoder_nxt_txn_dow = pad_sequences(txt_to_seq, maxlen=1, padding='pre')
# HOD
txt_to_seq = tokenizer_hod.texts_to_sequences(validation_universe.loc[index, ['NEXT_TXN_HOD']].tolist()) 
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

print("Ground Truth: ", target)
# print("Predicted: ", translation)
predicted_sentence = ', '.join(predicted_sentence)
print("Predicted: ", predicted_sentence)
