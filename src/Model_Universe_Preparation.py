# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:38:58 2020

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import pickle
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from functools import reduce


class Model_Universe():
    def __init__(self, wd, runType="model"):
        self.wd = wd
        os.chdir(self.wd)
        self.runType = runType.lower()
        self.raw_data = None
        self.model_universe = None
        self.pur_seq = None
        self.rtn_seq = None
        self.tfidf_df = None
        self.tf_idf_matrix = None

    def check_data(self, df, comment):
        print(comment)
        print(df.dtypes)
        print(df.shape)
        print("\n")
        
    def read_file(self, fileName):        
        self.raw_data = pd.read_csv(self.wd + fileName, converters={'Customer ID':str})
        self.check_data(df = self.raw_data, comment = "Read Raw Data File:")
        
    def preprocess_data(self, popularity_threshold=10, customer_threshold=1):
        self.raw_data = self.raw_data.dropna(axis=0, subset=['Customer ID', 'InvoiceDate', 'Invoice'])
        self.raw_data = self.raw_data[self.raw_data['Customer ID'] != ""]
        self.raw_data = self.raw_data.groupby(['Customer ID', 'InvoiceDate', 'Invoice', 'StockCode']).agg({'Quantity': sum, 'Price': sum}).reset_index(drop=False)
        self.raw_data['InvoiceDate'] = pd.to_datetime(self.raw_data['InvoiceDate'])
        
        # Dense rank for transactions
        self.raw_data['Tied_Invoices'] = self.raw_data.groupby(['InvoiceDate','Invoice']).ngroup()+1
        self.raw_data['Transaction_Rank'] = self.raw_data.groupby(['Customer ID'])['Tied_Invoices'].rank(method='dense', ascending=False)
        self.raw_data.drop(['Tied_Invoices'],axis=1,inplace=True)
        
        # Filtering products with less popularity
        Product_Popularity = self.raw_data.groupby(['StockCode'])['Customer ID'].count().reset_index(drop=False).rename(columns={'Customer ID':'Count'})
        Product_Popularity = Product_Popularity[Product_Popularity['Count']>popularity_threshold]
        self.raw_data = self.raw_data[self.raw_data['StockCode'].isin(Product_Popularity['StockCode'])]
        
        if self.runType == "model":
            # Filtering customers out who have only 1 transaction
            Transaction_Count = self.raw_data.groupby(['Customer ID'])['InvoiceDate'].nunique().reset_index(drop=False)
            Transaction_Count = Transaction_Count[Transaction_Count['InvoiceDate']>customer_threshold]
            self.raw_data = self.raw_data[self.raw_data['Customer ID'].isin(Transaction_Count['Customer ID'])]
            del Transaction_Count
            gc.collect()
        self.check_data(df = self.raw_data, comment = "Preprocessed Raw Data File:")
        
    def dv_creation(self, last_n_txns=1):
        if self.runType == "model":
            self.raw_data['DV'] = np.where(self.raw_data['Transaction_Rank']==last_n_txns, 1, 0)

        items = pd.DataFrame(self.raw_data['StockCode'].unique(), columns=['StockCode'])
        cids = pd.DataFrame(self.raw_data['Customer ID'].unique(), columns=['Customer ID'])
        items.loc[:,'constant'] = 1
        cids.loc[:,'constant'] = 1
        self.model_universe = pd.merge(cids, items, how='outer', on=['constant'])
        self.model_universe.drop('constant', axis=1, inplace=True)
        if self.runType == "model":
            self.model_universe = pd.merge(self.model_universe, self.raw_data.loc[self.raw_data['DV']==1,['Customer ID','StockCode', 'DV']], how="left", on=['Customer ID','StockCode']).fillna(0 ,inplace=False)            
            self.raw_data = self.raw_data[self.raw_data['DV'] != 1] # Exclude last txn
        self.check_data(df = self.model_universe, comment = "Created DV & Model Universe:")
            
    def downsample_df(self, downsample_pct=0.2):
        # Downsample Model Universe
        model_universe_majority = self.model_universe[self.model_universe['DV']==0]
        model_universe_minority = self.model_universe[self.model_universe['DV']==1]
         
        # Downsample majority class
        model_universe_majority_downsampled = resample(model_universe_majority, 
                                              replace=False,    # sample without replacement
                                              n_samples=int(len(model_universe_minority)/downsample_pct), # to match minority class
                                              random_state=123) # reproducible results
         
        # Combine minority class with downsampled majority class
        self.model_universe = pd.concat([model_universe_majority_downsampled, model_universe_minority])
        del model_universe_majority, model_universe_minority, model_universe_majority_downsampled
        gc.collect()
        
        # Include Customers and Products only in model universe
        self.raw_data = self.raw_data[(self.raw_data['Customer ID'].isin(self.model_universe['Customer ID'])) & (self.raw_data['StockCode'].isin(self.model_universe['StockCode']))]
        
        # Display new class counts
        self.check_data(df = self.model_universe, comment = "Downsampled Model Universe:")
        print("DV Count", self.model_universe.DV.value_counts())  
        
    def feature_creation(self):
        # TFIDF Features
        self.tfidf_df = self.raw_data.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID']).agg({
                'StockCode': lambda s: ', '.join(s)
                }).reset_index()
        
        if self.runType == "model":
            vectorizer = TfidfVectorizer()
            self.tf_idf_matrix = vectorizer.fit_transform(self.tfidf_df['StockCode'].values)
            pickle_byte_obj = [vectorizer]
            pickle.dump(pickle_byte_obj, open(self.wd + "tfidf.pkl", "wb"))
        else:
            tfidf_config = pickle.load(open(self.wd+"tfidf.pkl", "rb"))
            vectorizer = tfidf_config[0]
            self.tf_idf_matrix = vectorizer.transform(self.tfidf_df['StockCode'].values)
        
        print(self.tf_idf_matrix.shape)        
        self.tfidf_df.drop('StockCode', axis=1, inplace=True)
        self.check_data(df = self.tfidf_df, comment = "TFIDF:")
        
        # Last 3 Transactions       
        last_txn_rank = int(self.raw_data['Transaction_Rank'].min())
        txn_rank_1 = self.raw_data[self.raw_data['Transaction_Rank'] == last_txn_rank].sort_values(['InvoiceDate', 'StockCode'], ascending=(False, True)).groupby(['Customer ID']).agg({
                'StockCode': lambda s: ', '.join(s)
                }).reset_index().rename(columns={'StockCode':'txn_rank_1'})
        
        txn_rank_2 = self.raw_data[self.raw_data['Transaction_Rank'] == last_txn_rank+1].sort_values(['InvoiceDate', 'StockCode'], ascending=(False, True)).groupby(['Customer ID']).agg({
                'StockCode': lambda s: ', '.join(s)
                }).reset_index().rename(columns={'StockCode':'txn_rank_2'})
        
        txn_rank_3 = self.raw_data[self.raw_data['Transaction_Rank'] == last_txn_rank+2].sort_values(['InvoiceDate', 'StockCode'], ascending=(False, True)).groupby(['Customer ID']).agg({
                'StockCode': lambda s: ', '.join(s)
                }).reset_index().rename(columns={'StockCode':'txn_rank_3'})
        
        self.model_universe = reduce(lambda left,right: pd.merge(left,right,on='Customer ID',how="outer"), [self.model_universe, txn_rank_1, txn_rank_2, txn_rank_3]).fillna('' ,inplace=False)
        
        del txn_rank_1, txn_rank_2, txn_rank_3
        gc.collect()
        
        # Weekly Features
        self.raw_data['Week_Number'] = self.raw_data['InvoiceDate'].dt.strftime("%Y-%V")
        self.raw_data['Sequence_Number'] = self.raw_data.groupby(['Week_Number']).ngroup(ascending=False)+1
        self.raw_data.sort_values(['InvoiceDate'], ascending=(False), inplace=True)
        self.raw_data.drop('Week_Number', axis=1, inplace=True)
        
        self.pur_seq = self.raw_data[self.raw_data['Quantity']>0].groupby(['Customer ID','StockCode','Sequence_Number'])['Quantity'].sum().unstack('Sequence_Number', fill_value=0).reset_index(drop=False)
        self.rtn_seq = self.raw_data[self.raw_data['Quantity']<0].groupby(['Customer ID','StockCode','Sequence_Number'])['Quantity'].sum().unstack('Sequence_Number', fill_value=0).reset_index(drop=False)
        
        # Scaling features
        pur_seq_scaled_cols = [col for col in self.pur_seq.columns if col not in ['Customer ID','StockCode']]
        rtn_seq_scaled_cols = [col for col in self.rtn_seq.columns if col not in ['Customer ID','StockCode']]        
        if self.runType == "model":
            scaler_pur_seq = preprocessing.MinMaxScaler()
            scaler_rtn_seq = preprocessing.MinMaxScaler()
            self.pur_seq[pur_seq_scaled_cols] = scaler_pur_seq.fit_transform(self.pur_seq[pur_seq_scaled_cols])
            self.rtn_seq[rtn_seq_scaled_cols] = scaler_rtn_seq.fit_transform(self.rtn_seq[rtn_seq_scaled_cols])
            pickle_byte_obj = [scaler_pur_seq, scaler_rtn_seq]
            pickle.dump(pickle_byte_obj, open(self.wd + "scalers.pkl", "wb"))
        else:
            scaler_config = pickle.load(open(self.wd+"scalers.pkl", "rb"))
            scaler_pur_seq = scaler_config[0]
            scaler_rtn_seq = scaler_config[1]
            self.pur_seq[pur_seq_scaled_cols] = scaler_pur_seq.transform(self.pur_seq[pur_seq_scaled_cols])
            self.rtn_seq[rtn_seq_scaled_cols] = scaler_rtn_seq.transform(self.rtn_seq[rtn_seq_scaled_cols])
        
        self.check_data(df = self.pur_seq, comment = "Purchase Sequence:")
        self.check_data(df = self.rtn_seq, comment = "Return Sequence:")
        
    def get_model_universe(self, call_all_functions=True, **kwargs):
        if call_all_functions:
            self.read_file(kwargs['fileName'])
            self.preprocess_data(popularity_threshold = kwargs['popularity_threshold'], customer_threshold=kwargs['customer_threshold'])
            self.dv_creation(last_n_txns=kwargs['last_n_txns'])
            if self.runType == "model":
                self.downsample_df(downsample_pct=kwargs['downsample_pct'])
            self.feature_creation()
        return self.model_universe, self.pur_seq, self.rtn_seq, self.tfidf_df, self.tf_idf_matrix
