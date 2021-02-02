# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:13:37 2021

@author: kesha
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import preprocessing

from dataclasses import dataclass
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import re
import gc
import pickle
import os
from numpy.random import choice

wd = "C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Ta_Feng/Run_2/"
raw_data_loc = "C:/Users/kesha/Desktop/Projects/Next_Basket_Recommendations/Ta_Feng/"
fileName = "Train.csv"
runType = "model"

class data_to_arrays(object):
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
            if col in data.columns:
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
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_padding_length[col], padding='pre')
        return padded_docs

def str_concat(x):
    a = []
    tmp = [i for i in x] + ['eot']
    a.append(tmp)
    return a

# Partial Next TXN Prediction
def replace_elements_unique(x_list, t_list):
    if len(x_list)>0:
        x_list = x_list[0].copy()
        new_x = x_list.copy()
        n_size = int(np.ceil(len(x_list)*0.5))
        x_idx = choice(range(len(x_list)), size = n_size, replace=False)
        t_list = [x for x in t_list if x not in x_list] # Unseen products only
        t_idx = choice(range(len(t_list)), size = n_size, replace=False)
        for x, t in zip(x_idx, t_idx):
            new_x[x] = t_list[t]
        if 'eot' not in new_x:
            new_x.append('eot')
        return [new_x]
    else: return x_list

def split_dataframe(df, chunk_size = 1000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks 

def downsample_df(model_universe, dv_label, downsample_maj_pct=0.2, downsample_min_pct=1, downsample_min_ratio=None):
    # Downsample Model Universe
    model_universe_majority = model_universe[model_universe[dv_label]==0]
    model_universe_minority = model_universe[model_universe[dv_label]==1]
    
    if downsample_min_pct==1:
        model_universe_minority_downsampled = model_universe_minority
    else:
        model_universe_minority_downsampled = resample(model_universe_minority, 
                                              replace=False,    # sample without replacement
                                              n_samples=int(len(model_universe_minority)*downsample_min_pct),
                                              random_state=123) # reproducible results
        
    # Downsample majority class
    if downsample_min_ratio is not None:
        model_universe_majority_downsampled = resample(model_universe_majority, 
                                              replace=False,    # sample without replacement
                                              n_samples=int(len(model_universe_minority_downsampled)/downsample_min_ratio), # to match minority class
                                              random_state=123) # reproducible results
    else:    
        model_universe_majority_downsampled = resample(model_universe_majority, 
                                              replace=False,    # sample without replacement
                                              n_samples=int(len(model_universe_majority)*downsample_maj_pct),
                                              random_state=123) # reproducible results
    
    # Combine minority class with downsampled majority class
    model_universe = pd.concat([model_universe_majority_downsampled, model_universe_minority_downsampled])
    del model_universe_majority, model_universe_minority, model_universe_majority_downsampled, model_universe_minority_downsampled
    gc.collect()
  
    return model_universe


raw_data = pd.read_csv(raw_data_loc + fileName, converters={'Customer ID':str, 'StockCode': str})
raw_data = raw_data.dropna(axis=0, subset=['Customer ID', 'InvoiceDate'])
raw_data = raw_data[raw_data['Customer ID'] != ""]
raw_data = raw_data.groupby(['Customer ID', 'InvoiceDate', 'StockCode']).agg({'Quantity': sum}).reset_index(drop=False)
raw_data['InvoiceDate'] = pd.to_datetime(raw_data['InvoiceDate'])

# Transaction Rank
raw_data['Transaction_Rank'] = raw_data.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID'])['InvoiceDate'].rank(method='dense', ascending=False)

# Unique Customers and Items
list_of_items = raw_data['StockCode'].unique().tolist()
items = pd.DataFrame(raw_data['StockCode'].unique(), columns=['StockCode'])
items_list = ['','[UNK]',"sot","eot"] + raw_data['StockCode'].unique().tolist() + ['[mask]']
cids = pd.DataFrame(raw_data['Customer ID'].unique(), columns=['Customer ID'])
cids_list = ['','[UNK]'] + raw_data['Customer ID'].unique().tolist()

# Extract Last TXN
if runType == "model":
    dv_universe = raw_data[raw_data['Transaction_Rank']==1].copy()
    dv_universe.drop(['Transaction_Rank','Quantity'], axis=1, inplace=True)
    raw_data = raw_data[raw_data['Transaction_Rank']!=1]

    # Pre-Train Universe Creation
    pretrain_universe = raw_data.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID', 'InvoiceDate'])['StockCode'].apply(lambda x: str_concat(x)).reset_index(drop=False)
    # Cap Items Per TXN
    pretrain_universe['StockCode'] = pretrain_universe['StockCode'].apply(lambda x: [x[0][0:100]]) # Need to parameterize
    # ??
    # pretrain_universe["SHIFTED"] = pretrain_universe.groupby(['Customer ID'])['StockCode'].shift(1).apply(lambda d: d if isinstance(d, list) else [])
    
    pretrain_universe['TXN_HISTORY'] = pretrain_universe.groupby(['Customer ID'])['StockCode'].apply(lambda x: x.cumsum())
    # pretrain_universe.drop(["SHIFTED", "StockCode"], axis=1, inplace=True)
    # pretrain_universe['TXN_HISTORY'] = pretrain_universe.groupby(['Customer ID'])['StockCode'].apply(lambda x: x.cumsum())
    pretrain_universe.drop(["StockCode"], axis=1, inplace=True)
    
    # Extracting Most Recent TXN For Partial Next TXN Prediction
    pretrain_universe['TXN_1'] = pretrain_universe['TXN_HISTORY'].apply(lambda x: x[-1:])
    # Excluding Most Recent TXN from TXN_HISTORY
    pretrain_universe['TXN_HISTORY'] = pretrain_universe['TXN_HISTORY'].apply(lambda x: x[0:-1])
    # Reverse Order of TXNS so most recent comes first
    # pretrain_universe['TXN_HISTORY'] = pretrain_universe['TXN_HISTORY'].apply(lambda x: x[::-1])
    # Remove Rows Containing Empty TXN_HISTORY Values
    # pretrain_universe = pretrain_universe[pretrain_universe['TXN_HISTORY'].map(lambda d: len(d)) > 0]
    pretrain_universe['isNext'] = 1
    
    # Creating isNext 0 for Partial Next TXN Prediction
    partial_ntp = pretrain_universe[['Customer ID', 'InvoiceDate', 'TXN_HISTORY', 'TXN_1']].copy()
    partial_ntp['TXN_1'] = partial_ntp.apply(lambda x: replace_elements_unique(x['TXN_1'], list_of_items), axis=1)
    partial_ntp['isNext'] = 0
    
    # Concatenating both universes and downsample
    pretrain_universe = pd.concat([pretrain_universe, partial_ntp])
    del partial_ntp
    gc.collect()
    pretrain_universe.reset_index(drop=True, inplace=True)
    # pretrain_universe = downsample_df(model_universe = pretrain_universe, dv_label = 'isNext', downsample_maj_pct=0.5, downsample_min_pct=1, downsample_min_ratio=0.5)
    print(pretrain_universe['isNext'].value_counts())
    
    # Creating Strings
    pretrain_universe['TXN_HISTORY'] = pretrain_universe['TXN_HISTORY'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x)>0 else '')
    pretrain_universe['TXN_1'] = pretrain_universe['TXN_1'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x)>0 else '')
        
    # Start of TXN Token
    pretrain_universe['TXN_HISTORY'] = 'sot ' + pretrain_universe['TXN_HISTORY']
    pretrain_universe['TXN_1'] = 'sot ' + pretrain_universe['TXN_1']
    pretrain_universe['isNotNext'] = np.where(pretrain_universe['isNext']==1,0,1)


# Fine-Tuning Universe Creation
feature_universe = raw_data.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID', 'InvoiceDate'])['StockCode'].apply(lambda x: str_concat(x)).reset_index(drop=False)
# Cap Items Per TXN
feature_universe['StockCode'] = feature_universe['StockCode'].apply(lambda x: [x[0][0:100]]) # Need to parameterize

feature_universe['Transaction_Rank'] = feature_universe.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID'])['InvoiceDate'].rank(method='dense', ascending=False)
# Feature Creation
feature_universe['TXN_HISTORY'] = feature_universe.groupby(['Customer ID'])['StockCode'].apply(lambda x: x.cumsum())
# Dense Rank to Filter out Last TXN
feature_universe = feature_universe[feature_universe['Transaction_Rank']==1]
feature_universe.drop(['StockCode', 'Transaction_Rank', 'InvoiceDate'], axis=1, inplace=True)
# Reverse Order of TXNS so most recent comes first
# feature_universe['TXN_HISTORY'] = feature_universe['TXN_HISTORY'].apply(lambda x: x[::-1])
# Creating Strings
feature_universe['TXN_HISTORY'] = feature_universe['TXN_HISTORY'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x)>0 else '')
# Start of TXN Token
feature_universe['TXN_HISTORY'] = 'sot ' + feature_universe['TXN_HISTORY']


# Purchase Sequence Feature Creation
# if runType == "model":
#     # DV Universe Creation
#     dv_universe = pd.merge(dv_universe, feature_universe, on=["Customer ID"], how="left").fillna({"TXN_HISTORY": ''})
#     # Creating DV
#     dv_universe["DV"] = 1
#     # Transaction Rank
#     raw_data['Transaction_Rank'] = raw_data.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID'])['InvoiceDate'].rank(method='dense', ascending=False)

# pur_seq = raw_data[raw_data['Quantity']>0].groupby(['Customer ID', 'InvoiceDate', 'StockCode','Transaction_Rank'])['Quantity'].sum().unstack('Transaction_Rank', fill_value=0).reset_index(drop=False)
# pur_seq['Transaction_Rank'] = pur_seq.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID'])['InvoiceDate'].rank(method='dense', ascending=False)
# pur_seq = pur_seq[pur_seq['Transaction_Rank']==1]
# pur_seq.drop('Transaction_Rank', axis=1, inplace=True)
# gc.collect()

# # Scaling features
# pur_seq_scaled_cols = [col for col in pur_seq.columns if col not in ['Customer ID', 'InvoiceDate', 'StockCode']]       
# if runType == "model":
#     scaler_pur_seq = preprocessing.MinMaxScaler()
#     pur_seq[pur_seq_scaled_cols] = scaler_pur_seq.fit_transform(pur_seq[pur_seq_scaled_cols])
#     pickle_byte_obj = [scaler_pur_seq, pur_seq_scaled_cols]
#     pickle.dump(pickle_byte_obj, open(wd + "scalers.pkl", "wb"))
# else:
#     scaler_config = pickle.load(open(wd+"scalers.pkl", "rb"))
#     scaler_pur_seq = scaler_config[0]
#     pur_seq_scaled_cols = scaler_config[1]
#     for col in pur_seq_scaled_cols:
#         if col not in pur_seq.columns:
#             pur_seq[col] = 0
#     pur_seq[pur_seq_scaled_cols] = scaler_pur_seq.transform(pur_seq[pur_seq_scaled_cols])

# # Universe Creation
# feature_universe.drop_duplicates(inplace = True) 
# items.loc[:,'constant'] = 1
# feature_universe.loc[:,'constant'] = 1
# # Split Customer IDS into chunks for easier processing
# df_split = split_dataframe(cids, chunk_size=1000)

# print("Total Splits:", len(df_split))
# first_time = True
# for i, df in enumerate(df_split): 
#     print("Processing DF:", i)
#     combinations = pd.merge(feature_universe.loc[feature_universe["Customer ID"].isin(df["Customer ID"]), ["Customer ID", "constant"]], items, how='outer', on=['constant'])
#     combinations.drop('constant', axis=1, inplace=True)
#     if runType == "model":
#         universe = pd.merge(combinations, dv_universe.loc[dv_universe["Customer ID"].isin(combinations["Customer ID"]), ["Customer ID", "StockCode", "DV"]], on = ["Customer ID", "StockCode"], how="left").fillna({"DV": 0})
#         if first_time:            
#             model_universe = downsample_df(model_universe = universe, dv_label = "DV", downsample_maj_pct=0.2, downsample_min_pct=1, downsample_min_ratio=0.4)
#             first_time = False
#         else:            
#             tmp = downsample_df(model_universe = universe, dv_label = "DV", downsample_maj_pct=0.2, downsample_min_pct=1, downsample_min_ratio=0.4)
#             model_universe = pd.concat([model_universe, tmp])
#     else:
#         if first_time:
#             model_universe = pd.merge(combinations, feature_universe.loc[feature_universe["Customer ID"].isin(combinations["Customer ID"]), ["Customer ID"]], on = ["Customer ID"], how="left")
#             first_time = False
#         else:
#             tmp = pd.merge(combinations, feature_universe.loc[feature_universe["Customer ID"].isin(combinations["Customer ID"]), ["Customer ID"]], on = ["Customer ID"], how="left")
#             model_universe = pd.concat([model_universe, tmp])
            
# model_universe.reset_index(drop=True, inplace=True)
# feature_universe.drop('constant', axis=1, inplace=True)
# del combinations, tmp, universe
# gc.collect()



def tokenization(list_obj):
    t = tf.keras.preprocessing.text.Tokenizer(oov_token='[UNK]')
    word_index = dict()
    index_word = dict()
    for i, j in enumerate(list_obj):
        word_index[j] = i
        index_word[i] = j
    t.word_index = word_index
    t.index_word = index_word
    return t

if runType=="model":
    tokenizer_items = tokenization(items_list)
    txt_to_seq_items = tokenizer_items.texts_to_sequences(pretrain_universe['TXN_HISTORY'].tolist()) 
    padded_seq_items = pad_sequences(txt_to_seq_items, maxlen=300, padding='pre')
    txt_to_seq_items = tokenizer_items.texts_to_sequences(pretrain_universe['TXN_1'].tolist()) 
    padded_seq_txn_1 = pad_sequences(txt_to_seq_items, maxlen=300, padding='pre')
    print(tokenizer_items.word_index["sot"])
    print(tokenizer_items.word_index["eot"])
    print(len(tokenizer_items.word_index))
    
    tokenizer_cids = tokenization(cids_list)
    txt_to_seq_cids = tokenizer_cids.texts_to_sequences(pretrain_universe['Customer ID'].tolist()) 
    padded_seq_cids = pad_sequences(txt_to_seq_cids, maxlen=1, padding='pre')
    print(len(tokenizer_cids.word_index))

    # Get mask token id for masked language model
    mask_token_id = tokenizer_items.word_index["[mask]"]
    # Pickle the config and weights
    pickle.dump({'items': tokenizer_items,
                 'cids': tokenizer_cids}
                , open(wd+"tokenizer.pkl", "wb"))
else:
    pickle_byte_obj = pickle.load(open(wd+"tokenizer.pkl", "rb"))
    
    tokenizer_items = pickle_byte_obj['items']
    tokenizer_cids = pickle_byte_obj['cids']

if runType=="model":
    @dataclass
    class Config:
        MAX_LEN = 600
        BATCH_SIZE = 128
        FINE_TUNE_BATCH_SIZE = 64
        LR = 0.001
        ITEM_VOCAB_SIZE = int(len(tokenizer_items.word_index)) #30000
        CID_VOCAB_SIZE = int(len(tokenizer_cids.word_index))
        EMBED_DIM = 128
        NUM_HEAD = 8  # used in bert model
        FF_DIM = 128  # used in bert model
        NUM_LAYERS = 1
    
    config = Config()
    pickle_byte_obj = [config]
    pickle.dump(pickle_byte_obj, open(wd + "config.pkl", "wb"))

else:
    config = pickle.load(open(wd+"config.pkl", "rb"))[0]

def get_masked_input_and_labels(encoded_texts):
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 3] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights


# Create BERT model (Pretraining Model) for masked language modeling
def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


mlm_loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
pnsp_loss_fn = keras.losses.CategoricalCrossentropy()
# loss_fn = keras.losses.SparseCategoricalCrossentropy()
loss_tracker = tf.keras.metrics.Mean(name="loss")


class MaskedLanguageModel(tf.keras.Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:            
            mlm_predictions, pnsp_predictions = self([features], training=True)
            mlm_loss = mlm_loss_fn(labels['output_1'], mlm_predictions, sample_weight=sample_weight)
            pnsp_loss = pnsp_loss_fn(labels['output_2'], pnsp_predictions)
            loss = mlm_loss + pnsp_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]


def create_masked_language_bert_model():
    inputs_items = layers.Input((config.MAX_LEN,), dtype=tf.int64, name='input_1')
    inputs_segment = layers.Input((config.MAX_LEN,), dtype=tf.int64, name='segment')
    
    # Item Embeddings
    word_embeddings = layers.Embedding(
        config.ITEM_VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
    )(inputs_items)
    
    # Segment Embeddings
    segment_embeddings = layers.Embedding(
        3, config.EMBED_DIM, name="segment_embedding"
    )(inputs_segment)
    
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings + segment_embeddings

    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    # CIDS
    # inputs_cid = layers.Input((1,), dtype=tf.int64, name='input_2')
    # embedding_cid = layers.Embedding(config.CID_VOCAB_SIZE, config.EMBED_DIM, input_length=1, name='CID_embedding', trainable=True)(inputs_cid)
    # embedding_cid = layers.SpatialDropout1D(0.2)(embedding_cid)
    # all_layers = encoder_output + embedding_cid
    mlm_output = layers.Dense(config.ITEM_VOCAB_SIZE, name="output_1", activation="softmax")(
        encoder_output
    )
    
    encoder_output = layers.Flatten()(encoder_output)
    pnsp_output = layers.Dense(2, name="output_2", activation="softmax")(
        encoder_output
    )
    
    # mlm_model = MaskedLanguageModel(inputs=[inputs,inputs_cid], outputs=mlm_output, name="masked_bert_model")
    model = MaskedLanguageModel(inputs=[inputs_items, inputs_segment], outputs=[mlm_output, pnsp_output], name="masked_bert_model")

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    model.compile(optimizer=optimizer)
    return model

# id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
# token2id = {y: x for x, y in id2token.items()}


# class MaskedTextGenerator(keras.callbacks.Callback):
#     def __init__(self, sample_tokens, top_k=5):
#         self.sample_tokens = sample_tokens
#         self.k = top_k

#     def decode(self, tokens):
#         return " ".join([id2token[t] for t in tokens if t != 0])

#     def convert_ids_to_tokens(self, id):
#         return id2token[id]

#     def on_epoch_end(self, epoch, logs=None):
#         prediction = self.model.predict(self.sample_tokens)

#         masked_index = np.where(self.sample_tokens == mask_token_id)
#         masked_index = masked_index[1]
#         mask_prediction = prediction[0][masked_index]

#         top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
#         values = mask_prediction[0][top_indices]

#         for i in range(len(top_indices)):
#             p = top_indices[i]
#             v = values[i]
#             tokens = np.copy(sample_tokens[0])
#             tokens[masked_index[0]] = p
#             result = {
#                 "input_text": self.decode(sample_tokens[0].numpy()),
#                 "prediction": self.decode(tokens),
#                 "probability": v,
#                 "predicted mask token": self.convert_ids_to_tokens(p),
#             }
#             pprint(result)

if runType=="model":
    # sample_tokens = vectorize_layer(["00001823, sot 4714276145315, 20398576, 300410824006, [mask], 300410855611, 4710076132036, 4710085120628, 4710085172696, 4710085172702, eot"]) # 300410824105
    # generator_callback = MaskedTextGenerator(sample_tokens.numpy())
    
    bert_masked_model = create_masked_language_bert_model()
    print(bert_masked_model.summary())
    
    # Prepare data for masked language model
    x_masked_items, y_masked_labels, sample_weights = get_masked_input_and_labels(
        padded_seq_items
    )
    
    sample_weights_txn_1 = np.zeros(sample_weights.shape)
    items_segment = np.zeros(x_masked_items.shape)
    txn_1_segment = np.ones(padded_seq_txn_1.shape)
    
    # Combine TXN_HISTORY and TXN_1
    x_masked_items = np.concatenate((x_masked_items, padded_seq_txn_1), axis=1)
    y_masked_labels = np.concatenate((y_masked_labels, padded_seq_txn_1), axis=1)
    sample_weights = np.concatenate((sample_weights, sample_weights_txn_1), axis=1)
    segment = np.concatenate((items_segment, txn_1_segment), axis=1)
    y_pnsp_labels = pretrain_universe[['isNext','isNotNext']].values
    del sample_weights_txn_1, items_segment, txn_1_segment, padded_seq_txn_1, txt_to_seq_cids, txt_to_seq_items
    gc.collect()
    
    # mlm_ds = tf.data.Dataset.from_tensor_slices(
    #     ({"input_1": x_masked_items, "input_2": padded_seq_cids}, y_masked_labels, sample_weights)
    # )
    
    mlm_ds = tf.data.Dataset.from_tensor_slices(
        ({"input_1": x_masked_items, 'segment': segment}, {'output_1': y_masked_labels, 'output_2': y_pnsp_labels}, sample_weights)
    )


    mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE)
    
    # Train and Save
    # bert_masked_model.fit(dataset, steps_per_epoch=int(len(x_masked_items)//32)) #callbacks=[generator_callback]
    bert_masked_model.fit(mlm_ds, epochs=3)
    bert_masked_model.save(wd+"bert_mlm_nbr.h5")



# Fine-tune model

# Load pretrained bert model
mlm_model = keras.models.load_model(
    wd+"bert_mlm_nbr.h5", custom_objects={"MaskedLanguageModel": MaskedLanguageModel}
)
pretrained_bert_model = tf.keras.Model(
    mlm_model.input, mlm_model.get_layer("encoder_{0}/ffn_layernormalization".format(config.NUM_LAYERS-1)).output
)

# Train the weights?
pretrained_bert_model.trainable = True

top_k_categorical_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
    k=5, name="top_k_categorical_accuracy", dtype=None
)

def create_model(final_layer):
    
    inputs = []
    
    inputs_bert_1 = layers.Input((config.MAX_LEN,), dtype=tf.int64, name='input_1')
    # inputs_bert_2 = layers.Input((1,), dtype=tf.int64, name='input_2')
    # inputs_cid = layers.Input((1,), dtype=tf.int64, name='input_3')
    inputs.append(inputs_bert_1)
    # inputs.append(inputs_bert_2)
    # inputs.append(inputs_cid)
    sequence_output = pretrained_bert_model([inputs_bert_1])
    # cid_embedding = tf.keras.layers.Embedding(config.CID_VOCAB_SIZE, config.EMBED_DIM, input_length=1, name='CID_embedding', trainable=True)(inputs_cid)
    # pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    # Query-value attention of shape [batch_size, Tq, filters].
    # query_value_attention_seq = tf.keras.layers.Attention()([sequence_output, cid_embedding])
    pooled_output_bert = layers.Flatten()(sequence_output)
    # pooled_output_cid = layers.Flatten()(cid_embedding)
    # pooled_output = layers.Concatenate()([pooled_output_bert, pooled_output_cid])
    hidden_layer = layers.Dense(64, activation="relu")(pooled_output_bert)

    x = layers.BatchNormalization()(hidden_layer)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(final_layer, activation='softmax', name='Final_Layer')(x)   
 
    reco_model = Model(inputs, output, name='Recommendation_Model')        
    reco_model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=[top_k_categorical_accuracy])
    return reco_model

# Temporary
def dv_str_concat(x):
    a = []
    tmp = [i for i in x]
    a.append(tmp)
    return a

if runType == "model":
    dv_universe = dv_universe.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID', 'InvoiceDate'])['StockCode'].apply(lambda x: dv_str_concat(x)).reset_index(drop=False)
    dv_universe['StockCode'] = dv_universe['StockCode'].apply(lambda x : ' '.join(map(str, [i for j in x for i in j])) if len(x)>0 else '')
    dv_universe = pd.merge(dv_universe, feature_universe, on=["Customer ID"], how="left").fillna({"TXN_HISTORY": ''})

if runType == "model":
    # Tokenization for DV
    tokenizer_DV = tf.keras.preprocessing.text.Tokenizer(oov_token='')
    tokenizer_DV.fit_on_texts(dv_universe['StockCode'].astype(str))
    pickle_byte_obj = [tokenizer_DV]
    pickle.dump(pickle_byte_obj, open(wd + "tokenizer_DV.pkl", "wb"))

else:
    tokenizer_DV = pickle.load(open(wd+"tokenizer_DV.pkl", "rb"))[0]

if runType == "model":
    # Train-test split
    learn_ids, validation_ids = train_test_split(dv_universe, test_size = 0.1, random_state = 0)
    
    # IV
    txt_to_seq_items = tokenizer_items.texts_to_sequences(learn_ids['TXN_HISTORY'].tolist()) 
    padded_seq_items_learn = pad_sequences(txt_to_seq_items, maxlen=config.MAX_LEN, padding='pre')
    txt_to_seq_cids = tokenizer_cids.texts_to_sequences(learn_ids['Customer ID'].tolist()) 
    padded_seq_cids_learn = pad_sequences(txt_to_seq_cids, maxlen=1, padding='pre')
    # DV
    DV_learn = tokenizer_DV.texts_to_matrix(learn_ids['StockCode'].tolist(), mode='count')
    # txt_to_seq_items = tokenizer_DV.texts_to_sequences(learn_ids['StockCode'].tolist())
    # DV_learn = pad_sequences(txt_to_seq_items, maxlen=1, padding='post')
    
    # IV
    txt_to_seq_items = tokenizer_items.texts_to_sequences(validation_ids['TXN_HISTORY'].tolist()) 
    padded_seq_items_validation = pad_sequences(txt_to_seq_items, maxlen=config.MAX_LEN, padding='pre')
    txt_to_seq_cids = tokenizer_cids.texts_to_sequences(validation_ids['Customer ID'].tolist()) 
    padded_seq_cids_validation = pad_sequences(txt_to_seq_cids, maxlen=1, padding='pre')
    # DV
    DV_validation = tokenizer_DV.texts_to_matrix(validation_ids['StockCode'].tolist(), mode='count')
    # txt_to_seq_items = tokenizer_DV.texts_to_sequences(validation_ids['StockCode'].tolist())
    # DV_validation = pad_sequences(txt_to_seq_items, maxlen=1, padding='post')

    recommendation_model = create_model(final_layer = DV_learn.shape[1])
    print(recommendation_model.summary())
    reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=1, min_lr=1e-5,  verbose=1, mode = 'min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, restore_best_weights = True)
    callbacks_list = [reduce_learning_rate, early_stopping]
    
    
    train = tf.data.Dataset.from_tensor_slices(
            ({"input_1": padded_seq_items_learn}, DV_learn)
        )
    
    train = train.shuffle(1000).batch(config.FINE_TUNE_BATCH_SIZE)
    
    validation = tf.data.Dataset.from_tensor_slices(
            ({"input_1": padded_seq_items_validation}, DV_validation)
        )
    
    validation = validation.shuffle(1000).batch(config.FINE_TUNE_BATCH_SIZE)
    
    # Train and Save
    recommendation_model.fit(train, validation_data=validation, epochs=5, callbacks=[callbacks_list])
    recommendation_model.save(wd+"keras-recommender.h5")

else:
    # Scoring
    
    # IV
    txt_to_seq_items = tokenizer_items.texts_to_sequences(feature_universe['TXN_HISTORY'].tolist())
    padded_seq_items_test = pad_sequences(txt_to_seq_items, maxlen=500, padding='pre')
    txt_to_seq_cids = tokenizer_cids.texts_to_sequences(feature_universe['Customer ID'].tolist()) 
    padded_seq_cids_test = pad_sequences(txt_to_seq_cids, maxlen=1, padding='pre')
    
    recommendation_model = keras.models.load_model(
    wd+"keras-recommender.h5", custom_objects={"MaskedLanguageModel": MaskedLanguageModel}
    )

    test = tf.data.Dataset.from_tensor_slices(
            ({"input_1": padded_seq_items_test, "input_2": padded_seq_cids_test})
        )
    
    test = test.batch(config.FINE_TUNE_BATCH_SIZE)

    predictions = recommendation_model.predict(test, verbose=1)

    predictions = pd.DataFrame(predictions)
    feature_universe.reset_index(drop=True, inplace=True)
    predictions['Customer ID'] = feature_universe['Customer ID'].copy()
    
    def aggregate_recommendations(test_preds, id_label, top_k, chunk_size=10):
        test_preds['Sequence'] = np.arange(len(test_preds))
        test_preds['Sequence'] = test_preds['Sequence']/len(test_preds)
        test_preds['Chunk'] = pd.cut(test_preds['Sequence'], bins=chunk_size, labels=False)
        Chunks = test_preds['Chunk'].unique()
        test_preds.drop(['Sequence'], axis=1, inplace=True)
    
        first_time = True
        for Chunk in Chunks:
            print("Processing chunk:", Chunk)
            if first_time:
                tmp = test_preds[test_preds['Chunk']==Chunk].copy()
                tmp.drop(['Chunk'], axis=1, inplace=True)
                test_preds_df = pd.melt(tmp, id_vars=[id_label], var_name="Category", value_name="Probability")
                test_preds_df = test_preds_df.sort_values('Probability', ascending=False).groupby(id_label).head(top_k)
                test_preds_df["Rank"] = test_preds_df.groupby(id_label)["Probability"].rank(ascending=False)
                first_time = False
            else:
                tmp = test_preds[test_preds['Chunk']==Chunk].copy()
                tmp.drop(['Chunk'], axis=1, inplace=True)
                tmp = pd.melt(tmp, id_vars=[id_label], var_name="Category", value_name="Probability")
                tmp = tmp.sort_values('Probability', ascending=False).groupby(id_label).head(top_k)
                tmp["Rank"] = tmp.groupby(id_label)["Probability"].rank(ascending=False)
                test_preds_df = test_preds_df.append(tmp, ignore_index = True)
        return test_preds_df
    
    test_predictions = aggregate_recommendations(predictions, id_label="Customer ID", top_k=5, chunk_size=20)
    
    id2token = tokenizer_DV.index_word
    test_predictions = test_predictions.replace({"Category": id2token})
    test_predictions.rename(columns={"Category": "StockCode"}, inplace=True)
    test_predictions.rename(columns={"Probability": "Prediction"}, inplace=True)
    
    # Evaluation
    test_predictions['Rank'] = test_predictions.groupby(['Customer ID'])['Prediction'].rank(method='first', ascending=False)
    test_predictions = test_predictions[test_predictions['Rank']<=5]
    actuals = pd.read_csv(raw_data_loc + "Holdout.csv", converters={'Customer ID':str, 'StockCode': str, 'DV': int})
    
    print(len(test_predictions["Customer ID"].unique()))
    print(len(actuals["Customer ID"].unique()))
    
    actuals["DV"] = 1
    actuals.drop(["InvoiceDate", "Quantity"], axis=1, inplace=True)
    print(actuals.head())
    
    pred = pd.merge(test_predictions, actuals, on=["Customer ID", "StockCode"], how="left").fillna(0 ,inplace=False)
    pred['Recommendation'] = 1
    print(pred.head())
    print(len(pred["Customer ID"].unique()))
    
    pred = pred.groupby(['Customer ID']).agg({'Prediction': list, 'DV': list, 'Recommendation': list}).reset_index(drop=False)
    pred['DV_Count'] = pred.DV.str.len()
    pred = pred[pred['DV_Count']==5]
    from sklearn.metrics import ndcg_score
    true_relevance = np.asarray(pred['DV'].tolist())
    scores = np.asarray(pred['Prediction'].tolist())
    print("NDCG Score:",ndcg_score(true_relevance, scores))

    from sklearn.metrics import f1_score
    print("F1 Score:",f1_score(pred['DV'].tolist(), pred['Recommendation'].tolist(), average='weighted', zero_division=1))
    print(len(pred["Customer ID"].unique()))