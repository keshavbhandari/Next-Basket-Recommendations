# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:15:41 2020

@author: kbhandari
"""

import pandas as pd
import gc
import tensorflow as tf

from Autoencoder_Model import Autoencoder
from Model_Universe_Preparation import Model_Universe
from Recommendation_Engine import Recommendation_Engine
from Model_Evaluation import Evaluator
from Configs import downsample_pct, cat_max_seq_length, cont_max_seq_length, categorical_cols, autoencoder_batch_size, autoencoder_epochs, latent_features, recommendation_batch_size, recommendation_epochs, workers, wd, fileName, runType, runTest

    
if runType == "model":
    # Build Model Universe
    mdl_univ_object = Model_Universe(wd = wd,\
                                     runType="model")
    model_universe, pur_seq, rtn_seq, tfidf_df, tf_idf_matrix = \
    mdl_univ_object.get_model_universe(call_all_functions=True,\
                                       fileName = fileName,\
                                       popularity_threshold=10,\
                                       customer_threshold=1,\
                                       last_n_txns=1,\
                                       downsample_pct=downsample_pct)
    gc.collect()
    if runTest:
        model_universe = model_universe[0:10000]
    
    # Build Autoencoder Model
    autoencoder_model = Autoencoder(wd = wd, tf_idf_matrix = tf_idf_matrix, test_size = 0.2)
    autoencoder_model.create_model(latent_features = latent_features,\
                                  activation = "linear")
    autoencoder_model.fit_model(epochs = autoencoder_epochs,\
                                batch_size = autoencoder_batch_size,\
                                shuffle = True)
    
    gc.collect()
    # Loading a keras model
    from tensorflow.python.keras import backend as K
    session = tf.compat.v1.Session()
#    tf.compat.v1.reset_default_graph()
    graph = tf.compat.v1.get_default_graph()
    K.set_session(session)
    
    # Try this..
#    K.clear_session(). (import keras.backend as K)
    
    encoder = tf.keras.models.load_model(wd + "keras-encoder.h5")
    tf_idf_matrix = pd.DataFrame(data=encoder.predict(tf_idf_matrix.toarray()))
    gc.collect()
    
    # Build Recommendation Engine
    recommendation_object = Recommendation_Engine(wd = wd,\
                                                  model_universe = model_universe,\
                                                  categorical_cols = categorical_cols,\
                                                  cat_max_seq_length = cat_max_seq_length,\
                                                  cont_max_seq_length = cont_max_seq_length,\
                                                  pur_seq = pur_seq,\
                                                  rtn_seq = rtn_seq,\
                                                  tf_idf_matrix = tf_idf_matrix,\
                                                  tfidf_df = tfidf_df,\
                                                  encoder = encoder,\
                                                  latent_features = latent_features,\
                                                  run_type="model")
    
    learn_ids, validation_ids = recommendation_object.preprocess_data()
    recommendation_object.create_model()
    print(recommendation_object.reco_model.summary())
    recommendation_object.fit_model(batch_size=recommendation_batch_size,\
                                    epochs=recommendation_epochs,\
                                    workers=workers,\
                                    shuffle=True)
    gc.collect()
    
    # Evaluation    
    eval_object = Evaluator(wd = wd,\
                            sample_ids_dict = {'Learn': learn_ids, 'Validation': validation_ids},\
                            model_universe = model_universe,\
                            pur_seq = pur_seq,\
                            rtn_seq = rtn_seq,\
                            tf_idf_matrix = tf_idf_matrix,\
                            tfidf_df = tfidf_df,\
                            batch_size=1024)
    
    eval_object.get_predictions()
    eval_object.call_evaluation()
    
#    print("Train AUC:", roc_auc_score(ids_to_evaluate['Learn']['DV'], ids_to_evaluate['Learn']['Prediction']))
#    print("Validation AUC:",roc_auc_score(ids_to_evaluate['Validation']['DV'], ids_to_evaluate['Validation']['Prediction']))
    
#    model_hit_rate(df = ids_to_evaluate['Learn'])
#    model_hit_rate(df = ids_to_evaluate['Validation'])

# Scoring
else:
    
    mdl_univ_object = Model_Universe(wd = wd,\
                                     runType="scoring")
    model_universe, pur_seq, rtn_seq, tfidf_df, tf_idf_matrix = \
    mdl_univ_object.get_model_universe(call_all_functions=True,\
                                       fileName = fileName,\
                                       popularity_threshold=10,\
                                       customer_threshold=1,\
                                       last_n_txns=1,\
                                       downsample_pct=downsample_pct)
    gc.collect()
    
    # Encoder Scoring
    encoder = tf.keras.models.load_model(wd + "keras-encoder.h5")
    tf_idf_matrix = pd.DataFrame(data=encoder.predict(tf_idf_matrix.toarray()))
    gc.collect()
    
    if runTest:
        model_universe = model_universe[0:10000]
    recommendation_object = Recommendation_Engine(wd = wd,\
                                                  model_universe = model_universe,\
                                                  categorical_cols = None,\
                                                  cat_max_seq_length = None,\
                                                  cont_max_seq_length = None,\
                                                  pur_seq = pur_seq,\
                                                  rtn_seq = rtn_seq,\
                                                  tf_idf_matrix = tf_idf_matrix,\
                                                  tfidf_df = tfidf_df,\
                                                  encoder = None,\
                                                  latent_features = None,\
                                                  run_type="score")
    
    recommendation_object.preprocess_data()
    test_pred = recommendation_object.generate_predictions(test_ids = model_universe[['Customer ID', 'StockCode']],\
                                                           batch_size = recommendation_batch_size)
    test_pred.to_csv(wd + "Predictions.csv", header=True, index=False)