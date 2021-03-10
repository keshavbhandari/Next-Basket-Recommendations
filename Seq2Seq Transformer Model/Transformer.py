# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:09:13 2021

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
import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles( np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model = 512, num_heads = 8, causal=False, dropout=0.0):
    super(MultiHeadAttention, self).__init__()

    assert d_model % num_heads == 0
    depth = d_model // num_heads

    self.w_query = tf.keras.layers.Dense(d_model)
    self.split_reshape_query = tf.keras.layers.Reshape((-1,num_heads,depth))  
    self.split_permute_query = tf.keras.layers.Permute((2,1,3))      

    self.w_value = tf.keras.layers.Dense(d_model)
    self.split_reshape_value = tf.keras.layers.Reshape((-1,num_heads,depth))
    self.split_permute_value = tf.keras.layers.Permute((2,1,3))

    self.w_key = tf.keras.layers.Dense(d_model)
    self.split_reshape_key = tf.keras.layers.Reshape((-1,num_heads,depth))
    self.split_permute_key = tf.keras.layers.Permute((2,1,3))

    self.attention = tf.keras.layers.Attention(causal=causal, dropout=dropout)
    self.join_permute_attention = tf.keras.layers.Permute((2,1,3))
    self.join_reshape_attention = tf.keras.layers.Reshape((-1,d_model))

    self.dense = tf.keras.layers.Dense(d_model)

  def call(self, inputs, mask=None, training=None):
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v

    query = self.w_query(q)
    query = self.split_reshape_query(query)    
    query = self.split_permute_query(query)                 

    value = self.w_value(v)
    value = self.split_reshape_value(value)
    value = self.split_permute_value(value)

    key = self.w_key(k)
    key = self.split_reshape_key(key)
    key = self.split_permute_key(key)

    if mask is not None:
      if mask[0] is not None:
        mask[0] = tf.keras.layers.Reshape((-1,1))(mask[0])
        mask[0] = tf.keras.layers.Permute((2,1))(mask[0])
      if mask[1] is not None:
        mask[1] = tf.keras.layers.Reshape((-1,1))(mask[1])
        mask[1] = tf.keras.layers.Permute((2,1))(mask[1])

    attention = self.attention([query, value, key], mask=mask)
    attention = self.join_permute_attention(attention)
    attention = self.join_reshape_attention(attention)

    x = self.dense(attention)

    return x


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,  d_model = 512, num_heads = 8, dff = 2048, dropout = 0.0):
    super(EncoderLayer, self).__init__()

    self.multi_head_attention =  MultiHeadAttention(d_model, num_heads)
    self.dropout_attention = tf.keras.layers.Dropout(dropout)
    self.add_attention = tf.keras.layers.Add()
    self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(d_model)
    self.dropout_dense = tf.keras.layers.Dropout(dropout)
    self.add_dense = tf.keras.layers.Add()
    self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, mask=None, training=None):
    # print(mask)
    attention = self.multi_head_attention([inputs,inputs,inputs], mask = [mask,mask])
    attention = self.dropout_attention(attention, training = training)
    x = self.add_attention([inputs , attention])
    x = self.layer_norm_attention(x)
    # x = inputs

    ## Feed Forward
    dense = self.dense1(x)
    dense = self.dense2(dense)
    dense = self.dropout_dense(dense, training = training)
    x = self.add_dense([x , dense])
    x = self.layer_norm_dense(x)

    return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, txn_history_days_vocab_size, cid_vocab_size,
                 aisles_vocab_size, dept_vocab_size, dow_vocab_size, hod_vocab_size,
                 num_layers = 4, d_model = 512, num_heads = 8, dff = 2048, maximum_position_encoding = 10000, dropout = 0.0):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        
        self.embedding_items = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
        self.embedding_days = tf.keras.layers.Embedding(txn_history_days_vocab_size, d_model, mask_zero=True)
        self.embedding_cids = tf.keras.layers.Embedding(cid_vocab_size, d_model, mask_zero=True)
        # self.embedding_next_days = tf.keras.layers.Embedding(txn_history_days_vocab_size, d_model, mask_zero=True)
        self.embedding_aisles = tf.keras.layers.Embedding(aisles_vocab_size, d_model, mask_zero=True)
        self.embedding_dept = tf.keras.layers.Embedding(dept_vocab_size, d_model, mask_zero=True)
        self.embedding_dow = tf.keras.layers.Embedding(dow_vocab_size, d_model, mask_zero=True)
        self.embedding_hod = tf.keras.layers.Embedding(hod_vocab_size, d_model, mask_zero=True)

        self.pos = positional_encoding(maximum_position_encoding, d_model)
        
        self.encoder_layers = [ EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        w = self.embedding_items(inputs[0]) + self.embedding_days(inputs[1]) + self.embedding_aisles(inputs[4]) + self.embedding_dept(inputs[5]) + self.embedding_dow(inputs[6]) + self.embedding_hod(inputs[7])
        y = tf.concat([self.embedding_cids(inputs[2]), self.embedding_days(inputs[3]), self.embedding_dow(inputs[8]), self.embedding_hod(inputs[9])], axis=1)
        x = tf.keras.layers.Concatenate(axis=1)([y, w])
        # positional encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) 
        x += self.pos[: , :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        
        #Encoder layer
        mask_cids = self.embedding_cids.compute_mask(inputs[2])
        mask_items = self.embedding_items.compute_mask(inputs[0])
        mask_days = self.embedding_days.compute_mask(inputs[3])
        mask_dow = self.embedding_dow.compute_mask(inputs[8])
        mask_hod = self.embedding_hod.compute_mask(inputs[9])
        concatenated_mask = tf.concat([mask_cids, mask_days, mask_dow, mask_hod, mask_items], axis=1)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask = concatenated_mask)
        return x

    def compute_mask(self, inputs, mask=None):
        mask_cids = self.embedding_cids.compute_mask(inputs[2])
        mask_items = self.embedding_items.compute_mask(inputs[0])
        mask_days = self.embedding_days.compute_mask(inputs[3])
        mask_dow = self.embedding_dow.compute_mask(inputs[8])
        mask_hod = self.embedding_hod.compute_mask(inputs[9])
        concatenated_mask = tf.concat([mask_cids, mask_days, mask_dow, mask_hod, mask_items], axis=1)
        return concatenated_mask
    

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,  d_model = 512, num_heads = 8, dff = 2048, dropout = 0.0):
    super(DecoderLayer, self).__init__()

    self.multi_head_attention1 =  MultiHeadAttention(d_model, num_heads, causal = True)
    self.dropout_attention1 = tf.keras.layers.Dropout(dropout)
    self.add_attention1 = tf.keras.layers.Add()
    self.layer_norm_attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.multi_head_attention2 =  MultiHeadAttention(d_model, num_heads)
    self.dropout_attention2 = tf.keras.layers.Dropout(dropout)
    self.add_attention2 = tf.keras.layers.Add()
    self.layer_norm_attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(d_model)
    self.dropout_dense = tf.keras.layers.Dropout(dropout)
    self.add_dense = tf.keras.layers.Add()
    self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, mask=None, training=None):
    # print(mask)
    attention = self.multi_head_attention1([inputs[0],inputs[0],inputs[0]], mask = [mask[0],mask[0]])
    attention = self.dropout_attention1(attention, training = training)
    x = self.add_attention1([inputs[0] , attention])
    x = self.layer_norm_attention1(x)
    
    attention = self.multi_head_attention2([x, inputs[1],inputs[1]], mask = [mask[0],mask[1]])
    attention = self.dropout_attention2(attention, training = training)
    x = self.add_attention1([x , attention])
    x = self.layer_norm_attention1(x)


    ## Feed Forward
    dense = self.dense1(x)
    dense = self.dense2(dense)
    dense = self.dropout_dense(dense, training = training)
    x = self.add_dense([x , dense])
    x = self.layer_norm_dense(x)

    return x


class Decoder(tf.keras.layers.Layer):
  def __init__(self, target_vocab_size, num_layers = 4, d_model = 512, num_heads = 8, dff = 2048, maximum_position_encoding = 10000, dropout = 0.0):
    super(Decoder, self).__init__()

    self.d_model = d_model

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, mask_zero=True)
    self.pos = positional_encoding(maximum_position_encoding, d_model)

    self.decoder_layers = [ DecoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout)  for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs, mask=None, training=None):
    x = self.embedding(inputs[0])
    # positional encoding
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos[: , :tf.shape(x)[1], :]

    x = self.dropout(x, training=training)

    #Decoder layer
    embedding_mask = self.embedding.compute_mask(inputs[0])
    for decoder_layer in self.decoder_layers:
      x = decoder_layer([x,inputs[1]], mask = [embedding_mask, mask])

    return x

  # Comment this out if you want to use the masked_loss()
  def compute_mask(self, inputs, mask=None):
    return self.embedding.compute_mask(inputs[0])


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_model(config):
    # Hyperparameters
    num_layers = config['NUM_LAYERS']
    d_model = config['EMBED_DIM']
    dff = config['FF_DIM']
    num_heads = config['NUM_HEAD']
    dropout_rate = 0.1
    
    # Size of input vocab plus start and end tokens
    input_vocab_size = config['ITEM_VOCAB_SIZE']
    txn_days_vocab_size = config['DAYS_VOCAB_SIZE']
    target_vocab_size = config['ITEM_VOCAB_SIZE']
    cid_vocab_size = config['CID_VOCAB_SIZE']
    aisles_vocab_size = config['AISLES_VOCAB_SIZE']
    dept_vocab_size = config['DEPT_VOCAB_SIZE']
    dow_vocab_size = config['DOW_VOCAB_SIZE']
    hod_vocab_size = config['HOD_VOCAB_SIZE']
    
    items = tf.keras.layers.Input(shape=(None,), name = "input_1")
    target = tf.keras.layers.Input(shape=(None,), name = "input_2")
    days = tf.keras.layers.Input(shape=(None,), name = "input_3")
    cids = tf.keras.layers.Input(shape=(1,), name = "input_4")
    next_days = tf.keras.layers.Input(shape=(1,), name = "input_5")
    aisles = tf.keras.layers.Input(shape=(None,), name = "input_6")
    dept = tf.keras.layers.Input(shape=(None,), name = "input_7")
    dow = tf.keras.layers.Input(shape=(None,), name = "input_8")
    hod = tf.keras.layers.Input(shape=(None,), name = "input_9")
    next_dow = tf.keras.layers.Input(shape=(1,), name = "input_10")
    next_hod = tf.keras.layers.Input(shape=(1,), name = "input_11")
    all_encoder_inputs = [items, days, cids, next_days, aisles, dept, dow, hod, next_dow, next_hod]
    
    encoder = Encoder(input_vocab_size, txn_days_vocab_size, cid_vocab_size, aisles_vocab_size,
                      dept_vocab_size, dow_vocab_size, hod_vocab_size,
                      num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout_rate)
    decoder = Decoder(target_vocab_size, num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout_rate)
    
    x = encoder(all_encoder_inputs)
    x = decoder([target, x] , mask = encoder.compute_mask(all_encoder_inputs))
    x = tf.keras.layers.Dense(target_vocab_size, name="output_1")(x)
    
    model = tf.keras.models.Model(inputs=[items, target, days, cids, next_days, aisles, dept, dow, hod, next_dow, next_hod], outputs=x)
    
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    def masked_loss(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        _loss = loss(y_true, y_pred)
    
        mask = tf.cast(mask, dtype=_loss.dtype)
        _loss *= mask
          
        return tf.reduce_sum(_loss)/tf.reduce_sum(mask)

    metrics = [loss, masked_loss, tf.keras.metrics.SparseCategoricalAccuracy()]

    model.compile(optimizer=optimizer, loss = loss, metrics = metrics) # masked_
    
    print(model.summary())
    return model