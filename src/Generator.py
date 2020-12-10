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


class data_to_arrays():
    def __init__(self, cat_max_seq_length = 120, cont_max_seq_length = 120):
        self.cat_max_seq_length = cat_max_seq_length
        self.cont_max_seq_length = cont_max_seq_length
        super(data_to_arrays, self).__init__()

    # Continuous Sequence
    def get_continuous_sequence(self, list_of_dfs, cols_to_exclude):
        combined_arrays = None    
        first_time = True
        for i, df in enumerate(list_of_dfs):
            if len(cols_to_exclude)>0:
                df = df.drop(cols_to_exclude, axis=1, inplace=False)
            array = df.iloc[:,0:self.cont_max_seq_length].copy()
            if array.shape[1]>self.cont_max_seq_length:
                array = array.iloc[:,0:self.cont_max_seq_length]
            elif array.shape[1]<self.cont_max_seq_length:
                cols_to_add = self.cont_max_seq_length-array.shape[1]
                rows_to_add = array.shape[0]
                df = pd.DataFrame(np.zeros((rows_to_add,cols_to_add)))
                array = pd.concat([array, df], axis=1)
            array = np.array(array.astype(np.float))
            array = array.reshape(array.shape[0],array.shape[1],1)
            if first_time:
                combined_arrays = array
                first_time = False
            else:
                combined_arrays = np.concatenate((combined_arrays, array), -1)
        return combined_arrays
  
    def get_tokenizer(self, data, categorical_cols):
        word_index, word_tokenizer, max_length = {}, {}, {}
        for col in categorical_cols:
           print("Processing column:", col)        
           t = Tokenizer()
           t.fit_on_texts(data[col].astype(str))
           word_index[col] = t.word_index
           word_tokenizer[col] = t       
           max_length_value = max(data[col].str.split().str.len())
           max_length[col] = max_length_value if max_length_value < self.cat_max_seq_length else self.cat_max_seq_length
        return word_index, word_tokenizer, max_length
    
    def get_padding(self, data, categorical_cols, tokenizer, max_padding_length):
        padded_docs = {}
        for col in categorical_cols:
            t = tokenizer[col]
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))        
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_padding_length[col], padding='post')
        return padded_docs
  

class DataGenerator(data_to_arrays, keras.utils.Sequence):
    'Batch Generator for Keras'
    def __init__(self, sample_ids, dv_df, last_n_txn_seq, pur_seq, rtn_seq, tf_idf_matrix,\
                 tfidf_ids, autoencoder, categorical_cols, categorical_tokenizer, cat_max_seq_length,\
                 cont_max_seq_length, nrows, run_type = "model", batch_size=32, shuffle=True):        
        'Initialization'
        self.sample_ids = sample_ids[['Customer ID','StockCode']]
        self.dv_df = dv_df
        self.last_n_txn_seq = last_n_txn_seq
        self.pur_seq = pur_seq
        self.rtn_seq = rtn_seq
        self.tf_idf_matrix = tf_idf_matrix
        self.tfidf_ids = tfidf_ids
        self.autoencoder = autoencoder
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
        array_1 = pd.merge(batch_ids, self.pur_seq, on = ['Customer ID','StockCode'], how="left").fillna(0 ,inplace=False).drop(['Customer ID', 'StockCode'], axis=1, inplace=False)
        array_2 = pd.merge(batch_ids, self.rtn_seq, on = ['Customer ID','StockCode'], how="left").fillna(0 ,inplace=False).drop(['Customer ID', 'StockCode'], axis=1, inplace=False)
        sequence = super().get_continuous_sequence(list_of_dfs = [array_1, array_2], cols_to_exclude = [])
        
        # TF-IDF        
        tfidf_index = self.tfidf_ids[self.tfidf_ids['Customer ID'].isin(batch_ids['Customer ID'])].index    
#        sample_matrix = self.tf_idf_matrix[tfidf_index].toarray()
#        tfidf_df = pd.DataFrame(data=self.autoencoder.predict(sample_matrix))
        
        tfidf_df = self.tf_idf_matrix[self.tf_idf_matrix.index.isin(tfidf_index)].copy()
        
        tfidf_df['Customer ID'] = self.tfidf_ids.loc[tfidf_index, ['Customer ID']]
        tfidf_df_array = pd.merge(batch_ids['Customer ID'], tfidf_df, how="left", on=['Customer ID']).fillna(0, inplace=False).drop('Customer ID', axis=1, inplace=False)

        # Categorical Cols
        last_n_txn_seq_sample = pd.merge(batch_ids, self.last_n_txn_seq, on = ['Customer ID','StockCode'], how="left").fillna('', inplace=False)
        padded_docs = super().get_padding(data = last_n_txn_seq_sample,\
                       categorical_cols = self.categorical_cols,\
                       tokenizer = self.categorical_tokenizer,\
                       max_padding_length = self.cat_max_seq_length)
        
        input_list = []
        for col in self.categorical_cols:
            input_list.append(padded_docs[col])
        input_list.append(sequence)
        input_list.append(tfidf_df_array)
        
        if self.run_type.lower() == "model":
            target = pd.merge(batch_ids, self.dv_df, on = ['Customer ID','StockCode'], how="left").fillna(0 ,inplace=False).drop(['Customer ID', 'StockCode'], axis=1, inplace=False).values
            return input_list, target
        else:    
            return input_list