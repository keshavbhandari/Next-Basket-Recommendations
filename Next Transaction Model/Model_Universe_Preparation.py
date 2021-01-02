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
    def __init__(self, wd, od, runType="model"):
        self.wd = wd
        os.chdir(self.wd)
        self.od = od
        self.runType = runType.lower()
        self.raw_data = None
        self.dv_universe = None
        self.model_universe = None
        self.pur_seq = None
        self.tfidf_df = None
        self.tf_idf_matrix = None
        self.feature_universe = None

    def check_data(self, df, comment):
        print(comment)
        print(df.dtypes)
        print(df.shape)
        print("\n")
        
    def read_file(self, fileName):        
        self.raw_data = pd.read_csv(self.wd + fileName, converters={'Customer ID':str, 'StockCode': str})
        self.check_data(df = self.raw_data, comment = "Read Raw Data File:")
        
    def preprocess_data(self, popularity_threshold=10, customer_threshold=1, item_count=10):
        self.raw_data = self.raw_data.dropna(axis=0, subset=['Customer ID', 'InvoiceDate'])
        self.raw_data = self.raw_data[self.raw_data['Customer ID'] != ""]
        self.raw_data = self.raw_data.groupby(['Customer ID', 'InvoiceDate', 'StockCode']).agg({'Quantity': sum}).reset_index(drop=False)
        self.raw_data['InvoiceDate'] = pd.to_datetime(self.raw_data['InvoiceDate'])
        
        # Filtering products with less popularity
        Product_Popularity = self.raw_data.groupby(['StockCode'])['Customer ID'].count().reset_index(drop=False).rename(columns={'Customer ID':'Count'})
        Product_Popularity = Product_Popularity[Product_Popularity['Count']>popularity_threshold]
        self.raw_data = self.raw_data[self.raw_data['StockCode'].isin(Product_Popularity['StockCode'])]
        
#         if self.runType == "model":
#             # Filtering customers out who have only 1 transaction
#             Transaction_Count = self.raw_data.groupby(['Customer ID'])['InvoiceDate'].nunique().reset_index(drop=False)
#             Transaction_Count = Transaction_Count[Transaction_Count['InvoiceDate']>customer_threshold]
#             self.raw_data = self.raw_data[self.raw_data['Customer ID'].isin(Transaction_Count['Customer ID'])]
#             del Transaction_Count
#             gc.collect()
            
#             Item_Count = self.raw_data.groupby(['Customer ID'])['StockCode'].nunique().reset_index(drop=False)
#             Item_Count = Item_Count[Item_Count['StockCode']>=item_count]
#             self.raw_data = self.raw_data[self.raw_data['Customer ID'].isin(Item_Count['Customer ID'])]
        self.check_data(df = self.raw_data, comment = "Preprocessed Raw Data File:")
        
        # Dense rank for transactions
        self.raw_data['Transaction_Rank'] = self.raw_data.groupby(['Customer ID'])['InvoiceDate'].rank(method='dense', ascending=False)
    
    def downsample_df(self, model_universe, downsample_pct=0.2):
        # Downsample Model Universe
        model_universe_majority = model_universe[model_universe['DV']==0]
        model_universe_minority = model_universe[model_universe['DV']==1]
         
        # Downsample majority class
        model_universe_majority_downsampled = resample(model_universe_majority, 
                                              replace=False,    # sample without replacement
                                              n_samples=int(len(model_universe_minority)/downsample_pct), # to match minority class
                                              random_state=123) # reproducible results
         
        # Combine minority class with downsampled majority class
        model_universe = pd.concat([model_universe_majority_downsampled, model_universe_minority])
        del model_universe_majority, model_universe_minority, model_universe_majority_downsampled
        gc.collect()
  
        return model_universe
    
    def universe_creation(self, downsample_pct=0.3):
        
        def str_concat(x):
            a = []
            a.append([i for i in x])
            return a
        
        def split_dataframe(df, chunk_size = 1000): 
            chunks = list()
            num_chunks = len(df) // chunk_size + 1
            for i in range(num_chunks):
                chunks.append(df[i*chunk_size:(i+1)*chunk_size])
            return chunks 

        self.feature_universe = self.raw_data.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID', 'InvoiceDate'])['StockCode'].apply(lambda x: str_concat(x)).reset_index(drop=False)
        if self.runType == "model":
            self.feature_universe["SHIFTED"] = self.feature_universe.groupby(['Customer ID'])['StockCode'].shift(1).apply(lambda d: d if isinstance(d, list) else [])
        else:
            self.feature_universe["SHIFTED"] = self.feature_universe['StockCode']
        self.feature_universe['TXN_HISTORY'] = self.feature_universe.groupby(['Customer ID'])['SHIFTED'].apply(lambda x: x.cumsum())
        self.feature_universe.drop(["SHIFTED", "StockCode"], axis=1, inplace=True)
        
        # Feature Creation
        self.feature_universe['TXN_1'] = self.feature_universe['TXN_HISTORY'].apply(lambda x: x[-1:])
        self.feature_universe['TXN_2'] = self.feature_universe['TXN_HISTORY'].apply(lambda x: [x[-2]] if len(x)>=2 else [])
        self.feature_universe['TXN_3'] = self.feature_universe['TXN_HISTORY'].apply(lambda x: [x[-3]] if len(x)>=3 else [])
        
        self.feature_universe['TXN_HISTORY'] = self.feature_universe['TXN_HISTORY'].apply(lambda x : [i for j in x for i in j] if len(x)>0 else [])
        self.feature_universe['TXN_HISTORY'] = self.feature_universe['TXN_HISTORY'].apply(lambda x: ', '.join(map(str, x)))
        self.feature_universe['TXN_1'] = self.feature_universe['TXN_1'].apply(lambda x : [i for j in x for i in j] if len(x)>0 else [])
        self.feature_universe['TXN_1'] = self.feature_universe['TXN_1'].apply(lambda x: ', '.join(map(str, x)))
        self.feature_universe['TXN_2'] = self.feature_universe['TXN_2'].apply(lambda x : [i for j in x for i in j] if len(x)>0 else [])
        self.feature_universe['TXN_2'] = self.feature_universe['TXN_2'].apply(lambda x: ', '.join(map(str, x)))
        self.feature_universe['TXN_3'] = self.feature_universe['TXN_3'].apply(lambda x : [i for j in x for i in j] if len(x)>0 else [])
        self.feature_universe['TXN_3'] = self.feature_universe['TXN_3'].apply(lambda x: ', '.join(map(str, x)))
        
        self.dv_universe = pd.merge(self.raw_data, self.feature_universe, on=["Customer ID", "InvoiceDate"], how="inner")
        
        self.feature_universe['Transaction_Rank'] = self.feature_universe.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID'])['InvoiceDate'].rank(method='dense', ascending=False)
        self.dv_universe['Transaction_Rank'] = self.dv_universe.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID'])['InvoiceDate'].rank(method='dense', ascending=False)
        
        if self.runType == "model":
            self.dv_universe["DV"] = 1
            self.dv_universe["Quantity"] = self.dv_universe.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID', 'StockCode'])['Quantity'].shift(-1)
            self.dv_universe["Transaction_Rank"] = self.dv_universe.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID', 'StockCode'])['Transaction_Rank'].shift(-1)
        
        # Purchase Sequence Feature Creation
        self.pur_seq = self.dv_universe[self.dv_universe['Quantity']>0].groupby(['Customer ID', 'InvoiceDate', 'StockCode','Transaction_Rank'])['Quantity'].sum().unstack('Transaction_Rank', fill_value=0).reset_index(drop=False)
        self.pur_seq.rename(columns={x:y for x,y in zip(self.pur_seq.columns[3:],range(0,len(self.pur_seq.columns[3:])))}, inplace=True)

        # Scaling features
        pur_seq_scaled_cols = [col for col in self.pur_seq.columns if col not in ['Customer ID', 'InvoiceDate', 'StockCode']]       
        if self.runType == "model":
            scaler_pur_seq = preprocessing.MinMaxScaler()
            self.pur_seq[pur_seq_scaled_cols] = scaler_pur_seq.fit_transform(self.pur_seq[pur_seq_scaled_cols])
            pickle_byte_obj = [scaler_pur_seq, pur_seq_scaled_cols]
            pickle.dump(pickle_byte_obj, open(self.od + "scalers.pkl", "wb"))
        else:
            scaler_config = pickle.load(open(self.wd+"scalers.pkl", "rb"))
            scaler_pur_seq = scaler_config[0]
            pur_seq_scaled_cols = scaler_config[1]
            self.pur_seq[pur_seq_scaled_cols] = scaler_pur_seq.transform(self.pur_seq[pur_seq_scaled_cols])        
        self.check_data(df = self.pur_seq, comment = "Purchase Sequence:")
        
        # Scoring universe will have historic features only in last txn
        if self.runType == "model":
            self.feature_universe.drop('Transaction_Rank', axis=1, inplace=True)
            self.dv_universe.drop('Transaction_Rank', axis=1, inplace=True)
        else:
            self.feature_universe = self.feature_universe[self.feature_universe['Transaction_Rank'] == 1]
            self.dv_universe = self.dv_universe[self.dv_universe['Transaction_Rank'] == 1]
            self.feature_universe.drop('Transaction_Rank', axis=1, inplace=True)
            self.dv_universe.drop('Transaction_Rank', axis=1, inplace=True)
            
        # TF-IDF Creation
        if self.runType == "model":
            vectorizer = TfidfVectorizer()
            self.tfidf_df = self.feature_universe[["Customer ID", "InvoiceDate"]]
            self.tf_idf_matrix = vectorizer.fit_transform(self.feature_universe['TXN_HISTORY'].values)
            pickle_byte_obj = [vectorizer]
            pickle.dump(pickle_byte_obj, open(od + "tfidf.pkl", "wb"))
        else:
            tfidf_config = pickle.load(open(wd+"tfidf.pkl", "rb"))
            vectorizer = tfidf_config[0]
            self.tfidf_df = self.feature_universe[["Customer ID", "InvoiceDate"]]
            self.tf_idf_matrix = vectorizer.transform(self.feature_universe['TXN_HISTORY'].values)
        
        # Universe Creation
        items = pd.DataFrame(self.raw_data['StockCode'].unique(), columns=['StockCode'])
        unique_cids = pd.DataFrame(self.raw_data['Customer ID'].unique(), columns=['Customer ID'])
        self.feature_universe.drop_duplicates(inplace = True) 
        items.loc[:,'constant'] = 1
        self.feature_universe.loc[:,'constant'] = 1
        
        if self.runType == "model":
            df_split = split_dataframe(unique_cids, chunk_size=500)
        else:
            df_split = split_dataframe(unique_cids, chunk_size=5000)
        print("Total Splits:", len(df_split))
        first_time = True
        for i, df in enumerate(df_split): 
            print("Processing DF:", i)
            combinations = pd.merge(self.feature_universe.loc[self.feature_universe["Customer ID"].isin(df["Customer ID"]), ["Customer ID", "InvoiceDate", "constant"]], items, how='outer', on=['constant'])
            combinations.drop('constant', axis=1, inplace=True)
            if self.runType == "model":
                universe = pd.merge(combinations, self.dv_universe.loc[self.dv_universe["Customer ID"].isin(combinations["Customer ID"]), ["Customer ID", "InvoiceDate", "StockCode", "DV"]], on = ["Customer ID", "InvoiceDate", "StockCode"], how="left").fillna({"DV": 0})
                if first_time:
                    self.model_universe = self.downsample_df(model_universe = universe, downsample_pct = downsample_pct)
                    first_time = False
                else:
                    tmp = self.downsample_df(model_universe = universe, downsample_pct = downsample_pct)
                    self.model_universe = pd.concat([self.model_universe, tmp])
            else:
                if first_time:
                    self.model_universe = pd.merge(combinations, self.dv_universe.loc[self.dv_universe["Customer ID"].isin(combinations["Customer ID"]), ["Customer ID", "InvoiceDate", "StockCode"]], on = ["Customer ID", "InvoiceDate", "StockCode"], how="left")
                    first_time = False
                else:
                    tmp = pd.merge(combinations, self.dv_universe.loc[self.dv_universe["Customer ID"].isin(combinations["Customer ID"]), ["Customer ID", "InvoiceDate", "StockCode"]], on = ["Customer ID", "InvoiceDate", "StockCode"], how="left")
                    self.model_universe = pd.concat([self.model_universe, tmp])
        
        self.model_universe.reset_index(drop=True, inplace=True)
        self.feature_universe.drop('constant', axis=1, inplace=True)
        
        print("TFIDF Shape:", self.tf_idf_matrix.shape)
        self.check_data(df = self.feature_universe, comment = "Created Feature Universe:")
        self.check_data(df = self.model_universe, comment = "Created Model Universe:")
        
    def get_model_universe(self, call_all_functions=True, **kwargs):
        if call_all_functions:
            self.read_file(kwargs['fileName'])
            self.preprocess_data(popularity_threshold = kwargs['popularity_threshold'], customer_threshold=kwargs['customer_threshold'], item_count=kwargs['item_count'])
            self.universe_creation(downsample_pct=kwargs['downsample_pct'])
        return self.model_universe, self.pur_seq, self.tfidf_df, self.tf_idf_matrix, self.feature_universe
