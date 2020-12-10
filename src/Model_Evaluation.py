# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:12:32 2020

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from Recommendation_Engine import Recommendation_Engine

class Evaluator(Recommendation_Engine):
    def __init__(self, wd, sample_ids_dict, model_universe, pur_seq, rtn_seq, tf_idf_matrix,\
                 tfidf_df, batch_size):
        self.wd = wd
        self.sample_ids_dict = sample_ids_dict
        self.batch_size = batch_size
        self.ids_to_evaluate = {}
        
        super().__init__(wd = wd,\
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
        super().preprocess_data()

    def get_predictions(self):
        
        for df_name, ids in self.sample_ids_dict.items():
            test_pred = super().generate_predictions(test_ids = ids, batch_size=self.batch_size)
            self.ids_to_evaluate[df_name] = test_pred

    def model_hit_rate(self, df):
        preds_df = df.sort_values('Prediction', ascending=False).groupby('Customer ID').head(10)
        preds_df["Rank"] = preds_df.groupby("Customer ID")["Prediction"].rank(ascending=False)
        preds_df.drop('DV', axis=1, inplace=True)
        print("Processed predictions")
        
        actual_df = df.loc[df['DV']==1]
        print("Processed Actuals")
        
        for i in range(1, 11):
            hit_rate_df = pd.merge(actual_df, preds_df.loc[preds_df['Rank']<=i], how='left', on=['Customer ID','StockCode']).fillna(0)
            hit_rate = hit_rate_df.groupby('Customer ID').agg({'DV':'min', 'Rank':'max'})[['DV','Rank']].reset_index()
            hit_rate['Hit'] = np.where(hit_rate['Rank']>0, 1, 0)
            print("Hit Rate Top {0}: ".format(i), hit_rate['Hit'].sum() / len(hit_rate))
            
    def call_evaluation(self):
        for df_name, ids in self.ids_to_evaluate.items():
            print(df_name, "Hit Rate:")
            self.model_hit_rate(self.ids_to_evaluate[df_name])