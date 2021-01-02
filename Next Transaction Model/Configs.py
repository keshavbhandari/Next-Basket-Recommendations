# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:14:36 2020

@author: kbhandari
"""

# Configs
downsample_pct = 0.4
cat_max_seq_length = 50
cont_max_seq_length = 50 #85
categorical_cols = ['StockCode', 'TXN_1', 'TXN_2', 'TXN_3']
autoencoder_batch_size = 256
autoencoder_epochs = 10
latent_features = 100 #64
activation = "linear"
recommendation_batch_size = 4096
recommendation_scoring_batch_size = 24576 #49152
recommendation_epochs = 10
workers = 1
verbose = 2
wd = "/kaggle/input/tafeng-dataset/" #"/kaggle/input/next-basket-reco/"
od = "/kaggle/working/"
fileName = "Train.csv"
runType = "scoring" # model or scoring
runTest = False