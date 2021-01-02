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
from Generator import AutoencoderDataGenerator


# TFIDF AutoEncoder
# TFIDF AutoEncoder
class Autoencoder(AutoencoderDataGenerator):
    def __init__(self, wd, od, tfidf_ids, tf_idf_matrix, run_type,\
                 batch_size=32, test_size = 0.2, shuffle=True, verbose=1,\
                 latent_features=100, activation="linear"):
        self.wd = wd
        self.od = od
        self.tfidf_ids = tfidf_ids
        self.tf_idf_matrix = tf_idf_matrix
        self.run_type = run_type
        self.test_size = test_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.autoencoder, self.encoder = None, None
        self.latent_features = latent_features
        self.activation = activation
        if self.run_type == "model":
            self.tf_idf_learn, self.tf_idf_validation = train_test_split(self.tfidf_ids, test_size = self.test_size, random_state = 0)
        
    def create_model(self):
        input_shape = self.tf_idf_matrix.shape[1]
        output_shape = self.tf_idf_matrix.shape[1]

        input_data = layers.Input(shape=(input_shape,))
        encoded = layers.Dense(np.round(input_shape * 0.5), activation=self.activation, name='encoder_1')(input_data)
        encoded = layers.Dense(np.round(input_shape * 0.25), activation=self.activation, name='encoder_2')(encoded)
        encoded = layers.Dense(self.latent_features, activation=self.activation, activity_regularizer=tf.keras.regularizers.l1(10e-5), name='encoder_3')(encoded)

        decoded = layers.Dense(np.round(input_shape * 0.25), activation=self.activation, name='decoder_1')(encoded)
        decoded = layers.Dense(np.round(input_shape * 0.5), activation=self.activation, name='decoder_2')(decoded)
        decoded = layers.Dense(output_shape, activation=self.activation, name='decoder_3')(decoded)

        self.autoencoder = Model(input_data, decoded, name='autoencoder_model')
        print(self.autoencoder.summary())
        self.encoder = Model(input_data, encoded, name='encoder_model')
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        
    def fit_model(self, epochs = 20):
    
        train_data_gen = AutoencoderDataGenerator(\
                       tf_idf_matrix = self.tf_idf_matrix,\
                       tfidf_ids = self.tf_idf_learn,\
                       nrows=len(self.tf_idf_learn),\
                       run_type = "model",\
                       batch_size = self.batch_size,\
                       shuffle = self.shuffle)
        
        val_data_gen = AutoencoderDataGenerator(\
                       tf_idf_matrix = self.tf_idf_matrix,\
                       tfidf_ids = self.tf_idf_validation,\
                       nrows=len(self.tf_idf_validation),\
                       run_type = "model",\
                       batch_size = self.batch_size,\
                       shuffle = self.shuffle)
        
        reduce_learning_rate = ReduceLROnPlateau(monitor = "mse", factor = 0.6, patience=1, min_lr=1e-5,  verbose=1, mode = 'min')
        early_stopping = EarlyStopping(monitor='mse', mode='min', min_delta=0.00001, verbose=1, patience=3, restore_best_weights = True)
        callbacks_list = [reduce_learning_rate, early_stopping]
            
        self.autoencoder.fit_generator(generator=train_data_gen,
                                        validation_data=val_data_gen,
                                        epochs=epochs,
                                        shuffle=self.shuffle,
                                        verbose=self.verbose,
                                        callbacks=callbacks_list)
        
#         self.autoencoder.save_weights(self.od + "keras-autoencoder.h5", save_format='h5')
        self.encoder.save_weights(self.od + "keras-encoder.h5", save_format='h5')
        return True
    
    def generate_predictions(self, batch_size=1024):

        self.create_model()
        if self.run_type == "model":
            self.encoder.load_weights(self.od + "keras-encoder.h5")
        else:
            self.encoder.load_weights(self.wd + "keras-encoder.h5")
                        
        test_data_gen = AutoencoderDataGenerator(tf_idf_matrix = self.tf_idf_matrix,\
                       tfidf_ids = self.tfidf_ids,\
                       nrows=len(self.tfidf_ids),\
                       run_type = "scoring",\
                       batch_size = self.batch_size,\
                       shuffle = False)
        
        test_pred = self.encoder.predict_generator(test_data_gen, steps = np.ceil(len(self.tfidf_ids)/batch_size), verbose=self.verbose)
        test_pred = pd.DataFrame(data=test_pred)
        test_pred = pd.concat([self.tfidf_ids, test_pred], axis=1)
        return test_pred