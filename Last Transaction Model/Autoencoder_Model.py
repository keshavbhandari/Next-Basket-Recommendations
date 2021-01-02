# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:38:59 2020

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
from sklearn.model_selection import train_test_split


# TFIDF AutoEncoder
class Autoencoder():
    def __init__(self, wd, tf_idf_matrix, test_size = 0.2):
        self.wd = wd
        self.tf_idf_matrix = tf_idf_matrix
        self.test_size = test_size
        self.tf_idf_learn, self.tf_idf_validation = train_test_split(self.tf_idf_matrix, test_size = self.test_size, random_state = 0)
        self.autoencoder, self.encoder = None, None
        
    def create_model(self, latent_features = 100,\
                          activation = "linear"):
    
        input_shape = self.tf_idf_matrix.shape[1]
        output_shape = self.tf_idf_matrix.shape[1]
    
        input_data = layers.Input(shape=(input_shape,))
        encoded = layers.Dense(np.round(input_shape * 0.5), activation=activation, name='encoder_1')(input_data)
        encoded = layers.Dense(np.round(input_shape * 0.25), activation=activation, name='encoder_2')(encoded)
        encoded = layers.Dense(latent_features, activation=activation, name='encoder_3')(encoded)
        
        decoded = layers.Dense(np.round(input_shape * 0.25), activation=activation, name='decoder_1')(encoded)
        decoded = layers.Dense(np.round(input_shape * 0.5), activation=activation, name='decoder_2')(decoded)
        decoded = layers.Dense(output_shape, activation=activation, name='decoder_3')(decoded)
        
        self.autoencoder = Model(input_data, decoded, name='autoencoder_model')
        print(self.autoencoder.summary())
        self.encoder = Model(input_data, encoded, name='encoder_model')
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        
    def fit_model(self, epochs = 20, batch_size = 256, shuffle = True):
    
        reduce_learning_rate = ReduceLROnPlateau(monitor = "mse", factor = 0.6, patience=1, min_lr=1e-5,  verbose=1, mode = 'min')
        early_stopping = EarlyStopping(monitor='mse', mode='min', min_delta=0.0001, verbose=1, patience=3, restore_best_weights = True)
        callbacks_list = [reduce_learning_rate, early_stopping]
        
        self.autoencoder.fit(self.tf_idf_learn.toarray(), self.tf_idf_learn.toarray(),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        callbacks=callbacks_list,
                        validation_data=(self.tf_idf_validation.toarray(), self.tf_idf_validation.toarray()))
    
        self.autoencoder.save(self.wd + "keras-autoencoder.h5", include_optimizer=False)
        self.encoder.save(self.wd + "keras-encoder.h5", include_optimizer=False)
        return True