import tensorflow as tf
from layers.conv_lstm import GConvLSTMCell
from utils.graph import get_renormalized_adj_matrix
import numpy as np

def lstm_model(features, labels, mode, params):
    adj = np.load(params.adj)
    def sampling_probability_fn(step):
        def sigmoid_schedule(iteration, k, k0):
            return 1 - (k / (k + (tf.maximum(np.int64(0),iteration - k0) / k)))
        return sigmoid_schedule(step, 30, 3000)
    
    batch_size = tf.shape(features)[0]
    conv_matrix = tf.constant(get_renormalized_adj_matrix(adj-np.eye(params.n_nodes)), tf.float32) 
    with tf.variable_scope("input_output_projection"):
        W = tf.get_variable('projection', shape = [1, params.hidden_size], dtype=tf.float32)
        def projection_fn(x):
            x = tf.reshape(x, [-1, 1])
            x = tf.matmul(x, W)
            x = tf.reshape(x, [-1,params.n_nodes *params.hidden_size])
            return x
        
        def projection_transpose_fn(x):
            x = tf.reshape(x, [-1, params.hidden_size])
            x = tf.matmul(x, W, transpose_b=True)
            x = tf.reshape(x, [-1, params.n_nodes])
            return x

    encoder_cells = [GConvLSTMCell(params.hidden_size,activation = tf.nn.tanh, conv_matrix = conv_matrix,
                                use_residual_connection = True,
                                   projection_fn = projection_fn) for _ in range(params.n_layers)]
    encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, features, dtype=tf.float32)

    decoder_cells = [GConvLSTMCell(params.hidden_size,activation = tf.nn.tanh, conv_matrix = conv_matrix,
                                use_residual_connection = True,
                                   projection_fn = projection_fn) for _ in range(params.n_layers)]
    decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, params.n_nodes)
    decoder_cell._linear = projection_transpose_fn
    sequence_length = tf.tile([params.n_output_steps], [batch_size])
    if mode == tf.estimator.ModeKeys.TRAIN:
        zeros = tf.zeros([batch_size, 1, params.n_nodes], dtype=tf.float32)
        decoder_inputs = tf.concat([zeros, labels], axis=1)
        sampling_probability = sampling_probability_fn(tf.train.get_or_create_global_step())
        if (sampling_probability is not None
            and (tf.contrib.framework.is_tensor(sampling_probability)
             or sampling_probability > 0.0)):

            tf.summary.scalar("sampling_probability", sampling_probability)
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
            inputs=decoder_inputs,sequence_length=sequence_length, sampling_probability=sampling_probability)
        else:
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