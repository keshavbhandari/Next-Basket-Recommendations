# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:05:20 2020

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

from Generator import data_to_arrays, DataGenerator
from Attention_Layer import Attention
from Transformer_Layer import TransformerBlock, TokenAndPositionEmbedding


class Recommendation_Engine(data_to_arrays):
    def __init__(self, wd, model_universe, categorical_cols, cat_max_seq_length,\
                 cont_max_seq_length,pur_seq, rtn_seq, tf_idf_matrix, tfidf_df,\
                 encoder, latent_features, run_type = "model"):
        self.wd = wd
        self.model_universe = model_universe
        self.categorical_cols = categorical_cols
        self.cat_max_seq_length = cat_max_seq_length
        self.cont_max_seq_length = cont_max_seq_length
        self.pur_seq = pur_seq
        self.rtn_seq = rtn_seq
        self.tf_idf_matrix = tf_idf_matrix
        self.tfidf_df = tfidf_df
        self.encoder = encoder
        self.latent_features = latent_features
        self.learn_ids = None
        self.validation_ids = None
        self.test_ids = None
        self.word_index = None
        self.tokenizer = None
        self.cat_max_length = None
        self.run_type = run_type
        self.reco_model = None
        super().__init__(cat_max_seq_length = self.cat_max_seq_length, cont_max_seq_length = self.cont_max_seq_length)
        
    def preprocess_data(self):
        if self.run_type.lower() == "model":
            # Train-test split
            self.learn_ids, self.validation_ids = train_test_split(self.model_universe[['Customer ID','StockCode','DV']], test_size = 0.2, random_state = 0)        
            self.word_index, self.tokenizer, self.cat_max_length = super().get_tokenizer(data = self.model_universe, categorical_cols = self.categorical_cols)             
            pickle_byte_obj = [self.word_index, self.tokenizer, self.cat_max_length, self.cat_max_seq_length, self.cont_max_seq_length, self.categorical_cols, self.latent_features]
            pickle.dump(pickle_byte_obj, open(self.wd + "data_config.pkl", "wb"))            
            return self.learn_ids, self.validation_ids        
        else:
            data_config = pickle.load(open(self.wd+"data_config.pkl", "rb"))
            self.word_index, self.tokenizer, self.cat_max_length, self.cat_max_seq_length, self.cont_max_seq_length, self.categorical_cols, self.latent_features = data_config[0], data_config[1], data_config[2], data_config[3], data_config[4], data_config[5], data_config[6]            
            
    def create_model(self):
    
        inputs = []
        all_layers = []
        
        # Categorical Cols
        for col in self.categorical_cols:
            vocab_size = len(self.word_index[col]) + 1
            embed_size = int(np.min([np.ceil((vocab_size)/2), 30])) #25
            input_cat_cols = layers.Input(shape=(self.cat_max_length[col],))
            inputs.append(input_cat_cols)
            if self.cat_max_length[col] > 30:
                embedding = TokenAndPositionEmbedding(self.cat_max_length[col], vocab_size, embed_size)(input_cat_cols)
                embedding = TransformerBlock(embed_size, num_heads=1, ff_dim=32, rate=0.2)(embedding)
            else:
                embedding = layers.Embedding(vocab_size, embed_size, input_length=self.cat_max_length[col], name='{0}_embedding'.format(col), trainable=True)(input_cat_cols)
                embedding = layers.SpatialDropout1D(0.2)(embedding)
            embedding = layers.Flatten()(embedding)
            all_layers.append(embedding)
        
    #   Dense layer for continous variable sequence
        input_cont_seq_cols = layers.Input(shape=(self.cont_max_seq_length,2))
        inputs.append(input_cont_seq_cols)
    
        continuous_sequence = layers.Bidirectional(layers.LSTM(self.cont_max_seq_length, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(input_cont_seq_cols)
#        continuous_sequence = layers.TimeDistributed(layers.Dense(25, activation='relu'))(continuous_sequence)
#        continuous_sequence = layers.Flatten()(continuous_sequence)
        continuous_sequence=Attention()(continuous_sequence)
        all_layers.append(continuous_sequence)
        
    #   Dense layer for TF-IDF
        input_tfidf = layers.Input(shape=(self.latent_features,))
        inputs.append(input_tfidf)
    
        numeric = layers.BatchNormalization()(input_tfidf)
        numeric = layers.Dense(50, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(numeric) #50
        numeric = layers.Dropout(.4)(numeric)
        numeric = layers.Dense(25, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(numeric) #30  
        numeric = layers.Dropout(.2)(numeric)
        all_layers.append(numeric)
    
        x = layers.Concatenate()(all_layers)
    
        x = layers.BatchNormalization()(x)
        x = layers.Dense(50, activation="relu")(x)
        x = layers.Dropout(.2)(x)
        output = layers.Dense(1, activation='sigmoid', name='Final_Layer')(x)   
     
        self.reco_model = Model(inputs, output, name='Recommendation_Model')        
        self.reco_model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=['accuracy'])
        
    def fit_model(self, batch_size=256, epochs=5, workers=1, shuffle=True):
        
        train_data_gen = DataGenerator(sample_ids = self.learn_ids,\
                       dv_df = self.model_universe[['Customer ID','StockCode','DV']],\
                       last_n_txn_seq = self.model_universe,\
                       pur_seq = self.pur_seq,\
                       rtn_seq = self.rtn_seq,\
                       tf_idf_matrix = self.tf_idf_matrix,\
                       tfidf_ids = self.tfidf_df,\
                       autoencoder = self.encoder,\
                       categorical_cols = self.categorical_cols,\
                       categorical_tokenizer = self.tokenizer,\
                       cat_max_seq_length = self.cat_max_length,\
                       cont_max_seq_length = self.cont_max_seq_length,\
                       nrows=len(self.learn_ids),\
                       run_type = "model",\
                       batch_size = batch_size,\
                       shuffle = shuffle)
        
        val_data_gen = DataGenerator(sample_ids = self.validation_ids,\
                       dv_df = self.model_universe[['Customer ID','StockCode','DV']],\
                       last_n_txn_seq = self.model_universe,\
                       pur_seq = self.pur_seq,\
                       rtn_seq = self.rtn_seq,\
                       tf_idf_matrix = self.tf_idf_matrix,\
                       tfidf_ids = self.tfidf_df,\
                       autoencoder = self.encoder,\
                       categorical_cols = self.categorical_cols,\
                       categorical_tokenizer = self.tokenizer,\
                       cat_max_seq_length = self.cat_max_length,\
                       cont_max_seq_length = self.cont_max_seq_length,\
                       nrows=len(self.validation_ids),\
                       run_type = "model",\
                       batch_size = batch_size,\
                       shuffle = shuffle)
        
        reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=1, min_lr=1e-5,  verbose=1, mode = 'min')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, verbose=1, patience=3, restore_best_weights = True)
#        filepath="C:/My Desktop/Projects/model_checkpoint.hdf5"
#        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [reduce_learning_rate, early_stopping]
        
        recommendation = self.reco_model.fit_generator(generator=train_data_gen,
                                        validation_data=val_data_gen,
                                        epochs=epochs,
                                        shuffle=False,
                                        verbose=1,
                                        callbacks=callbacks_list,
                                        use_multiprocessing=False,
                                        workers=workers
                                        )
        recommendation.model.save_weights(self.wd + 'keras-recommender.h5', save_format='h5')
#        recommendation.model.save(self.wd + "keras-recommender.h5", include_optimizer=False)
        
    def generate_predictions(self, test_ids=None, batch_size=256):
        if test_ids is None:
                if "DV" in self.model_universe.columns:
                    self.test_ids = self.model_universe[['Customer ID','StockCode','DV']]
                else:
                    self.test_ids = self.model_universe[['Customer ID','StockCode']]  
        else:
            self.test_ids = test_ids.reset_index(drop=True, inplace=False)

        self.encoder = tf.keras.models.load_model(self.wd + "keras-encoder.h5")
        self.create_model()
        self.reco_model.load_weights(self.wd + "keras-recommender.h5")
                        
        test_data_gen = DataGenerator(sample_ids = self.test_ids,\
                       dv_df = None,\
                       last_n_txn_seq = self.model_universe,\
                       pur_seq = self.pur_seq,\
                       rtn_seq = self.rtn_seq,\
                       tf_idf_matrix = self.tf_idf_matrix,\
                       tfidf_ids = self.tfidf_df,\
                       autoencoder = self.encoder,\
                       categorical_cols = self.categorical_cols,\
                       categorical_tokenizer = self.tokenizer,\
                       cat_max_seq_length = self.cat_max_length,\
                       cont_max_seq_length = self.cont_max_seq_length,\
                       nrows=len(self.test_ids),\
                       run_type = "score",\
                       batch_size = batch_size,\
                       shuffle = False)
        
        test_pred = self.reco_model.predict_generator(test_data_gen, steps = np.ceil(len(self.test_ids)/batch_size), verbose=1)
        test_pred = pd.DataFrame(test_pred, columns = ['Prediction'])
        test_pred = pd.concat([self.test_ids, test_pred], axis=1)
        return test_pred
