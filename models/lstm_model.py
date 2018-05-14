import tensorflow as tf
from layers.conv_gru import GCGRU
from utils.graph import get_renormalized_adj_matrix
import numpy as np

def lstm_model(features, labels, mode, params):

    batch_size = tf.shape(features)[0]

    encoder_cells = [tf.contrib.rnn.BasicLSTMCell(params.hidden_size,
            activation = tf.nn.tanh) for _ in range(params.n_layers)]
    encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, features, dtype=tf.float32)

    decoder_cells = [tf.contrib.rnn.BasicLSTMCell(params.hidden_size,
            activation = tf.nn.tanh) for _ in range(params.n_layers)]
    decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, params.n_nodes)
    sequence_length = tf.tile([params.n_output_steps], [batch_size])
    if mode == tf.estimator.ModeKeys.TRAIN:
        zeros = tf.zeros([batch_size, 1, params.n_nodes], dtype=tf.float32)
        decoder_inputs = tf.concat([zeros, labels], axis=1)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, sequence_length)
                               
    else:
        helper = tf.contrib.seq2seq.InferenceHelper(tf.identity, [params.n_nodes], tf.float32,
                                                    tf.zeros([batch_size, params.n_nodes], dtype = tf.float32),
                                                    lambda sample_ids : tf.cast(tf.zeros([batch_size]), tf.bool))
                               
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=decoder_cell, helper=helper, initial_state = tuple(encoder_states))
                               
    decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder, impute_finished=False, maximum_iterations = params.n_output_steps)
    return decoder_outputs[0]