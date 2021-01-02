# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:38:59 2020

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class AutoencoderDataGenerator(keras.utils.Sequence):
    'Batch Generator for Keras'
    def __init__(self, tf_idf_matrix, tfidf_ids, nrows, run_type = "model", batch_size=32, shuffle=True):        
        'Initialization'
        self.tf_idf_matrix = tf_idf_matrix
        self.tfidf_ids = tfidf_ids
        self.run_type = run_type
        self.nrows = nrows
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nrows / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'        
#        print(index)
        # Generate indexes of the batch
        batch = self.tfidf_ids[index*self.batch_size:(index+1)*self.batch_size]
        if self.run_type.lower() == "model":
            X, Y = self.data_preprocess(batch_ids = batch)        
            return X,Y
        else:
            X = self.data_preprocess(batch_ids = batch) 
            return X

    def on_epoch_end(self):
        if self.shuffle:
            self.tfidf_ids = self.tfidf_ids.sample(frac=1)
        if self.run_type.lower() == "model":
            print("Data Generator: Epoch End")
    
    def data_preprocess(self, batch_ids):
        'Processes data in batch_size samples'
        
        # TF-IDF   
        tfidf_index = pd.merge(self.tfidf_ids, batch_ids, on=['Customer ID', 'InvoiceDate'], how="inner").index        
        tfidf_df_sample = self.tf_idf_matrix[tfidf_index].copy()
                
        if self.run_type.lower() == "model":
            return tfidf_df_sample.toarray(), tfidf_df_sample.toarray()
        else:    
            return tfidf_df_sample.toarray()


class DataGenerator(data_to_arrays, keras.utils.Sequence):
    'Batch Generator for Keras'
    def __init__(self, sample_ids, dv_df, last_n_txn_seq, pur_seq, tf_idf_matrix,\
                 tfidf_ids, categorical_cols, categorical_tokenizer, cat_max_seq_length,\
                 cont_max_seq_length, nrows, run_type = "model", batch_size=32, shuffle=True):        
        'Initialization'
        self.sample_ids = sample_ids[['Customer ID','InvoiceDate','StockCode']]
        self.dv_df = dv_df
        self.last_n_txn_seq = last_n_txn_seq
        self.pur_seq = pur_seq
        self.tf_idf_matrix = tf_idf_matrix
        self.tfidf_ids = tfidf_ids
        self.categorical_cols = categorical_cols
        self.categorical_tokenizer = categorical_tokenizer
        self.cat_max_seq_length = cat_max_seq_length
        self.cont_max_seq_length = cont_max_seq_length
        self.run_type = run_type
        self.nrows = nrows
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        super().__init__(cat_max_seq_length = self.cat_max_seq_length, cont_max_seq_length = self.cont_max_seq_length)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nrows / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'        
#        print(index)
        # Generate indexes of the batch
        batch = self.sample_ids[index*self.batch_size:(index+1)*self.batch_size]
        if self.run_type.lower() == "model":
            X, Y = self.data_preprocess(batch_ids = batch)        
            return X,Y
        else:
            X = self.data_preprocess(batch_ids = batch) 
            return X

    def on_epoch_end(self):
        if self.shuffle:
            self.sample_ids = self.sample_ids.sample(frac=1)
        if self.run_type.lower() == "model":
            print("Data Generator: Epoch End")
    
    def data_preprocess(self, batch_ids):
        'Processes data in batch_size samples'
        
        # Weekly Sequence
        array_1 = pd.merge(batch_ids, self.pur_seq, on = ['Customer ID','InvoiceDate','StockCode'], how="left").fillna(0 ,inplace=False).drop(['Customer ID','InvoiceDate','StockCode'], axis=1, inplace=False)
        sequence = super().get_continuous_sequence(list_of_dfs = [array_1], cols_to_exclude = [])
        
        # TF-IDF
        tfidf_df_array = pd.merge(batch_ids[['Customer ID', 'InvoiceDate']], self.tf_idf_matrix, how="left", on=['Customer ID', 'InvoiceDate']).fillna(0, inplace=False).drop(['Customer ID', 'InvoiceDate'], axis=1, inplace=False)

        # Categorical Cols
        last_n_txn_seq_sample = pd.merge(batch_ids, self.last_n_txn_seq, on = ['Customer ID','InvoiceDate'], how="left").fillna('', inplace=False)
        padded_docs = super().get_padding(data = last_n_txn_seq_sample,\
                       categorical_cols = self.categorical_cols,\
                       tokenizer = self.categorical_tokenizer,\
                       max_padding_length = self.cat_max_seq_length)
        
        input_list = []
        for col in self.categorical_cols:
            input_list.append(padded_docs[col])
        input_list.append(sequence)
        input_list.append(tfidf_df_array.values)
        
        if self.run_type.lower() == "model":
            target = pd.merge(batch_ids, self.dv_df, on = ['Customer ID','InvoiceDate','StockCode'], how="left").fillna(0 ,inplace=False).drop(['Customer ID', 'InvoiceDate', 'StockCode'], axis=1, inplace=False).values
            return input_list, target
        else:    
            return input_list