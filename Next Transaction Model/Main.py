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
from Configs import *

    
if runType == "model":
    # Build Model Universe
    mdl_univ_object = Model_Universe(wd = wd,\
                                     od = od,\
                                     runType="model")
    model_universe, pur_seq, tfidf_df, tf_idf_matrix, txn_sequence = \
    mdl_univ_object.get_model_universe(call_all_functions=True,\
                                       fileName = fileName,\
                                       popularity_threshold=9,\
                                       customer_threshold=1,\
                                       item_count=9,\
                                       downsample_pct=downsample_pct)
    gc.collect()
    if runTest:
        model_universe = model_universe[0:10000]
    
    # Build Autoencoder Model
    autoencoder_model = Autoencoder(wd = wd,\
                                    od = od,\
                                    tfidf_ids = tfidf_df,\
                                    tf_idf_matrix = tf_idf_matrix,\
                                    run_type = runType,\
                                    batch_size=128,\
                                    test_size = 0.2,\
                                    shuffle=True,\
                                    verbose=verbose,\
                                    latent_features=latent_features,\
                                    activation=activation)
    autoencoder_model.create_model()
    autoencoder_model.fit_model(epochs = autoencoder_epochs)
    
    # Predictions
    autoencoder_object = Autoencoder(wd = wd,\
                                     od = od,\
                                    tfidf_ids = tfidf_df,\
                                    tf_idf_matrix = tf_idf_matrix,\
                                    run_type = "scoring",\
                                    batch_size=128,\
                                    test_size = 0.2,\
                                    shuffle=False,\
                                    verbose=verbose,\
                                    latent_features=latent_features,\
                                    activation=activation)
    tf_idf_predictions = autoencoder_object.generate_predictions(batch_size=1024)
    gc.collect()
    
    # Build Recommendation Engine
    recommendation_object = Recommendation_Engine(wd = wd,\
                                                  od = od,\
                                                  model_universe = model_universe,\
                                                  txn_sequence = txn_sequence,\
                                                  categorical_cols = categorical_cols,\
                                                  cat_max_seq_length = cat_max_seq_length,\
                                                  cont_max_seq_length = cont_max_seq_length,\
                                                  pur_seq = pur_seq,\
                                                  tf_idf_matrix = tf_idf_predictions,\
                                                  tfidf_df = tfidf_df,\
                                                  latent_features = latent_features,\
                                                  run_type="model")
    
    learn_ids, validation_ids = recommendation_object.preprocess_data()
    recommendation_object.create_model()
    print(recommendation_object.reco_model.summary())
    recommendation_object.fit_model(batch_size=recommendation_batch_size,\
                                    epochs=recommendation_epochs,\
                                    workers=workers,\
                                    shuffle=True,\
                                    verbose=verbose)
    gc.collect()
    
    # Evaluation 
    pred_learn = recommendation_object.generate_predictions(test_ids = learn_ids, batch_size = recommendation_batch_size, verbose=verbose)
    pred_validation = recommendation_object.generate_predictions(test_ids = validation_ids, batch_size = recommendation_batch_size, verbose=verbose)

    
#    eval_object = Evaluator(wd = wd,\
#                            sample_ids_dict = {'Learn': learn_ids, 'Validation': validation_ids},\
#                            model_universe = model_universe,\
#                            txn_sequence = txn_sequence,\
#                            pur_seq = pur_seq,\
#                            tf_idf_matrix = tf_idf_predictions,\
#                            tfidf_df = tfidf_df,\
#                            batch_size=8192)
#    
#    eval_object.get_predictions()
#    eval_object.call_evaluation()
    
#    pred = eval_object.ids_to_evaluate
#    pred_validation = pred['Validation']
    pred_validation['Rank'] = pred_validation.groupby(['Customer ID'])['Prediction'].rank(method='dense', ascending=False)
    pred_validation = pred_validation[pred_validation['Rank']<=5]
    pred_validation = pred_validation.groupby(['Customer ID']).agg({'Prediction': list, 'DV': list}).reset_index(drop=False)
    pred_validation['DV_Count'] = pred_validation.DV.str.len()
    pred_validation = pred_validation[pred_validation['DV_Count']==5]
    from sklearn.metrics import ndcg_score
    true_relevance = np.asarray(pred_validation['DV'].tolist())
    scores = np.asarray(pred_validation['Prediction'].tolist())
    print(ndcg_score(true_relevance, scores))
    
#    pred_learn = pred['Learn']
    pred_learn['Rank'] = pred_learn.groupby(['Customer ID'])['Prediction'].rank(method='dense', ascending=False)
    pred_learn = pred_learn[pred_learn['Rank']<=5]
    pred_learn = pred_learn.groupby(['Customer ID']).agg({'Prediction': list, 'DV': list}).reset_index(drop=False)
    pred_learn['DV_Count'] = pred_learn.DV.str.len()
    pred_learn = pred_learn[pred_learn['DV_Count']==5]
    from sklearn.metrics import ndcg_score
    true_relevance = np.asarray(pred_learn['DV'].tolist())
    scores = np.asarray(pred_learn['Prediction'].tolist())
    print(ndcg_score(true_relevance, scores))
    
    print("Train AUC:", roc_auc_score(pred['Learn']['DV'], pred['Learn']['Prediction']))
    print("Validation AUC:",roc_auc_score(pred['Validation']['DV'], pred['Validation']['Prediction']))
    
#    model_hit_rate(df = ids_to_evaluate['Learn'])
#    model_hit_rate(df = ids_to_evaluate['Validation'])

# Scoring
else:
    
    mdl_univ_object = Model_Universe(wd = wd,\
                                     od = od,\
                                     runType="scoring")
    model_universe, pur_seq, tfidf_df, tf_idf_matrix, txn_sequence = \
    mdl_univ_object.get_model_universe(call_all_functions=True,\
                                       fileName = fileName,\
                                       popularity_threshold=9,\
                                       customer_threshold=1,\
                                       item_count=9,\
                                       last_n_txns=1,\
                                       downsample_pct=downsample_pct)
    del mdl_univ_object
    gc.collect()
    
    # Predictions
    autoencoder_object = Autoencoder(wd = wd,\
                                     od = od,\
                                    tfidf_ids = tfidf_df,\
                                    tf_idf_matrix = tf_idf_matrix,\
                                    run_type = "scoring",\
                                    batch_size=256,\
                                    test_size = 0.2,\
                                    shuffle=False,\
                                    verbose=verbose,\
                                    latent_features=latent_features,\
                                    activation=activation)
    tf_idf_predictions = autoencoder_object.generate_predictions(batch_size=1024)
    del autoencoder_object
    gc.collect()
    
    if runTest:
        model_universe = model_universe[0:10000]
    recommendation_object = Recommendation_Engine(wd = wd,\
                                                  od = od,\
                                                  model_universe = model_universe,\
                                                  txn_sequence = txn_sequence,\
                                                  categorical_cols = None,\
                                                  cat_max_seq_length = None,\
                                                  cont_max_seq_length = None,\
                                                  pur_seq = pur_seq,\
                                                  tf_idf_matrix = tf_idf_predictions,\
                                                  tfidf_df = tfidf_df,\
                                                  latent_features = None,\
                                                  run_type="score")
    
    recommendation_object.preprocess_data()
    
    def split_dataframe(df, chunk_size = 1000): 
        chunks = list()
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
        return chunks 

    cids = pd.DataFrame(model_universe['Customer ID'].unique(), columns=['Customer ID'])
    df_split = split_dataframe(cids, chunk_size=1000)

    test_pred = None
    first_time = True
    for i, df in enumerate(df_split): 
        print("Processing DF:", i, "out of", len(df_split))
        if first_time:
            test_pred = recommendation_object.generate_predictions(test_ids = model_universe.loc[model_universe["Customer ID"].isin(df["Customer ID"]), ['Customer ID', 'InvoiceDate', 'StockCode']],\
                                                                   batch_size = recommendation_scoring_batch_size, verbose=verbose)
            test_pred['Rank'] = test_pred.groupby(['Customer ID'])['Prediction'].rank(method='first', ascending=False)
            test_pred = test_pred[test_pred['Rank']<=5]
            first_time = False
        else:
            tmp_pred = recommendation_object.generate_predictions(test_ids = model_universe.loc[model_universe["Customer ID"].isin(df["Customer ID"]), ['Customer ID', 'InvoiceDate', 'StockCode']],\
                                                                   batch_size = recommendation_scoring_batch_size, verbose=verbose)
            tmp_pred['Rank'] = tmp_pred.groupby(['Customer ID'])['Prediction'].rank(method='first', ascending=False)
            tmp_pred = tmp_pred[tmp_pred['Rank']<=5]
            test_pred = pd.concat([test_pred, tmp_pred])
            del tmp_pred
            gc.collect()
            
    del model_universe
    gc.collect()
    print(test_pred.head())
    print(test_pred.shape)
    test_pred.to_csv(od + "Predictions.csv", header=True, index=False)
    
    # Evaluation
    actuals = pd.read_csv(wd+ "Holdout.csv", converters={'Customer ID':str, 'StockCode': str})
    actuals["DV"] = 1
    actuals.drop(["InvoiceDate", "Quantity", "Transaction_Rank"], axis=1, inplace=True)
    print(actuals.head())
    
    test_pred = pd.merge(test_pred, actuals, on=["Customer ID", "StockCode"], how="left").fillna(0 ,inplace=False)
    print(test_pred.head())
    
    test_pred = test_pred.groupby(['Customer ID']).agg({'Prediction': list, 'DV': list}).reset_index(drop=False)
    test_pred['DV_Count'] = test_pred.DV.str.len()
    test_pred = test_pred[test_pred['DV_Count']==5]
    from sklearn.metrics import ndcg_score
    true_relevance = np.asarray(test_pred['DV'].tolist())
    scores = np.asarray(test_pred['Prediction'].tolist())
    print("NDCG Score:",ndcg_score(true_relevance, scores))
    
    test_pred['Prediction'] = 1
    from sklearn.metrics import f1_score
    print("F1 Score:",f1_score(test_pred['DV'].tolist(), test_pred['Prediction'].tolist(), average='binary', zero_division=1))

    
    print(len(test_pred['Customer ID'].unique()))