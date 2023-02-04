# encoding: utf-8

import numpy as np
import tensorflow as tf
import os

# tf functions
class Saver(object):
    def __init__(self, sess, save_dir, max_to_keep=10):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.sess = sess
        self.save_dir = save_dir
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=max_to_keep)

    def save(self, step):
        self.saver.save(self.sess, self.save_dir, global_step=step)

    def restore(self, idx=''):
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        model_path = self.save_dir+idx if idx else ckpt.model_checkpoint_path # 'dir/-110'
        print("Reading model parameters from %s" % model_path)
        self.saver.restore(self.sess, model_path)

def get_weight_varible(name, shape):
        return tf.compat.v1.get_variable(name, initializer=tf.random.uniform(shape, -0.01, 0.01))

def getmask(length, max_len, out_shape):
    ''' 
    length shape:[batch_size]
    '''
    ret = tf.cast(tf.sequence_mask(length, max_len), tf.float32)
    return tf.reshape(ret, out_shape)

def biLSTM(inputs,length,n_hidden,scope):
    ''' 
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )
    
    return tf.concat(outputs, 2)

def LSTM(inputs,length,n_hidden,scope):
    ''' 
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden]
    '''
    outputs, state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )
    
    return outputs


def softmax_by_length(inputs, length):
    ''' 
    input shape:[batch_size, 1, max_len]
    length shape:[batch_size]
    return shape:[batch_size, 1, max_len]
    '''
    inputs = tf.exp(tf.cast(inputs, tf.float32))
    inputs *= getmask(length, tf.shape(inputs)[2], tf.shape(inputs))
    _sum = tf.reduce_sum(inputs, reduction_indices=2, keepdims =True) + 1e-9
    return inputs / _sum

def att_var(inputs,length,w1,b1,w2):
    ''' 
    input shape:[batch_size, max_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len, n_hidden = (tf.shape(inputs)[1], tf.shape(inputs)[2])
    tmp = tf.reshape(inputs, [-1, n_hidden])
    u = tf.tanh(tf.matmul(tmp, w1) + b1)
    alpha = tf.reshape(tf.matmul(u, w2), [-1, 1, max_len])
    alpha = softmax_by_length(alpha, length)
    return tf.reshape(tf.matmul(alpha, inputs), [-1, n_hidden]) 

def layer_normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs

def multihead_attention(queries,
                        keys,
                        values,
                        units_query,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention"):
    ''' 
    queries/keys/values shape:[batch_size, max_len, n_hidden]
    outputs shape:[batch_size, max_len, n_hidden]
    '''
    with tf.compat.v1.variable_scope(scope):
        # Linear projections
        Q = tf.layers.dense(queries, units_query, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, units_query, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(values, units_query, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5) #  QK^T / (d_k)^0.5

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        # the corresponding element in the output should be taken from paddings (if true) or outputs (if false)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # (h*N, T_q, T_k)

        # Dropouts
        outputs = tf.layers.dropout(
            outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    return outputs


def multihead_attention_realtime(queries,
                        keys,
                        values,
                        units_query,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention"):
    ''' 
    queries/keys/values shape:[batch_size, max_len, n_hidden]
    outputs shape:[batch_size, max_len, n_hidden]
    '''
    with tf.compat.v1.variable_scope(scope):
        # Linear projections
        Q = tf.layers.dense(queries, units_query, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, units_query, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(values, units_query, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5) #  QK^T / (d_k)^0.5

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        # print("\nattention时不考虑当前utt之后的utt...\n")
        real_time_mask = tf.linalg.band_part(tf.ones((tf.shape(queries)[1], tf.shape(keys)[1])),-1,0)
        key_masks *= real_time_mask

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        # the corresponding element in the output should be taken from paddings (if true) or outputs (if false)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        query_masks *= real_time_mask
        outputs *= query_masks  # (h*N, T_q, T_k)


        # Dropouts
        outputs = tf.layers.dropout(
            outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    return outputs


def pw_feedforward(inputs, num_units, out_units):
    u1 = tf.layers.dense(inputs, num_units, use_bias=True, activation=tf.nn.relu)  # (N, T_q, C)
    outputs = tf.layers.dense(u1, out_units, use_bias=True)
    return outputs

def standard_trans(inputs, n_hidden, n_head=1, scope="standard_trans"):
    ''' 
    inputs/outputs shape:[batch_size, max_len, n_hidden]
    '''
    z_hat = multihead_attention(queries=inputs, keys=inputs, values=inputs, units_query=n_hidden, num_heads=n_head, dropout_rate=0, is_training=True, scope=scope)
    z = layer_normalize(z_hat+inputs)
    outputs_hat = pw_feedforward(z, n_hidden, n_hidden)
    outputs = layer_normalize(outputs_hat+z)
    return outputs

# for real-time conversation
def standard_trans_realtime(inputs, n_hidden, n_head=1, scope="standard_trans"):
    ''' 
    inputs/outputs shape:[batch_size, max_len, n_hidden]
    '''
    z_hat = multihead_attention_realtime(queries=inputs, keys=inputs, values=inputs, units_query=n_hidden, num_heads=n_head, dropout_rate=0, is_training=True, scope=scope)
    z = layer_normalize(z_hat+inputs)
    outputs_hat = pw_feedforward(z, n_hidden, n_hidden)
    outputs = layer_normalize(outputs_hat+z)
    return outputs

def CR_2Dtrans(inputs, n_hidden, n_head=1, scope="CR_2Dtrans"):
    ''' 
    inputs/outputs shape:[batch_size, max_len, max_len, n_hidden]
    '''
    
    def row_2Datt(inputs, name='row'):
        max_len = tf.shape(inputs)[1]
        z_row = tf.reshape(inputs, [-1, max_len, n_hidden])
        z_row = multihead_attention(queries=z_row, keys=z_row, values=z_row, units_query=n_hidden, num_heads=n_head, dropout_rate=0, is_training=True, scope=scope + name)
        return tf.reshape(z_row, [-1, max_len, max_len, n_hidden])
    
    def col_2Datt(inputs, name='col'):
        z_col = tf.transpose(inputs, [0, 2, 1, 3])
        z_col = row_2Datt(z_col, name=name)
        z_col = tf.transpose(z_col, [0, 2, 1, 3])
        return z_col

    def CR_2Datt(inputs, name='CR_2Datt_'):
        z_row = row_2Datt(inputs, name=name+'row')
        z_col = col_2Datt(inputs, name=name+'col')
        outputs = (z_row + z_col)/2.
        return outputs

    z_hat = CR_2Datt(inputs, name=scope+'CR_2Datt_')
    z = layer_normalize(z_hat+inputs)
    outputs_hat = pw_feedforward(z, n_hidden, n_hidden)
    outputs = layer_normalize(outputs_hat+z)
    return outputs

def cross_modal_trans_ori(Q, KV, n_hidden, n_head=1, cmt_dropout=0, scope="cross_modal_trans_ori"):
    ''' 
    inputs/outputs shape:[batch_size, max_len, n_hidden]
    '''
    Q, KV = map(lambda x: layer_normalize(x), [Q, KV])
    z_hat = multihead_attention(queries=Q, keys=KV, values=KV, units_query=n_hidden, num_heads=n_head, dropout_rate=cmt_dropout, is_training=True, scope=scope)
    z = layer_normalize(z_hat+Q)
    outputs_hat = pw_feedforward(z, n_hidden, n_hidden)
    outputs = layer_normalize(outputs_hat+z)
    return outputs

def cross_modal_trans(Q, KV, n_hidden, cmt_num_layer=4, n_head=1, cmt_dropout=0, scope="cross_modal_trans"):
    ''' 
    inputs/outputs shape:[batch_size, max_len, n_hidden]
    '''
    z_hat_last = Q
    for ii in range(cmt_num_layer):
        z_hat = multihead_attention(queries=layer_normalize(z_hat_last), keys=layer_normalize(KV), values=layer_normalize(KV), units_query=n_hidden, num_heads=n_head, dropout_rate=cmt_dropout, is_training=True, scope=scope+'_att{}'.format(ii+1))
        z_hat = z_hat + layer_normalize(z_hat_last)
        z_hat_last = z_hat

    outputs_hat = pw_feedforward(layer_normalize(z_hat), n_hidden, n_hidden)
    outputs = outputs_hat + layer_normalize(z_hat)
    return outputs
