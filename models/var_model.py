import tensorflow as tf
from layers.conv_gru import GCGRU
from utils.graph import get_renormalized_adj_matrix
import numpy as np

def var_model(features, labels, mode, params):

    batch_size = tf.shape(features)[0]

    W = tf.get_variable('weights', [params.n_nodes*params.n_input_steps, params.n_nodes], dtype = tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('bias', [params.n_nodes], 
                        dtype = tf.float32, initializer = tf.zeros_initializer())
    outputs = []
    X = features
    for i in range(12):
        result = []
        outputs.append(tf.matmul(tf.reshape(X,[-1,params.n_nodes * params.n_input_steps]), W) + b)
        if i!= 11:
            if mode == tf.estimator.ModeKeys.TRAIN:
                new_input = labels[:,i]
            else:
                new_input = outputs[-1]
            X = tf.concat([X[:,1:,:], tf.expand_dims(new_input,1)], axis=1)
    outputs = tf.stack(outputs, 1)
    return outputs