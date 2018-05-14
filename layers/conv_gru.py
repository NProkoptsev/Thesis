from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
from layers.graph_convolution import chebnet

class GCGRU(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units,
                 forget_bias=1.0,
                 activation=None, reuse=None, conv_matrix=None,
                 kernel_initializer=None, bias_initializer=tf.zeros_initializer()):
        super(GCGRU, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh
        self.conv_matrix = conv_matrix
        self.K = int(conv_matrix.get_shape()[0])
        self.n_nodes = int(conv_matrix.get_shape()[1])
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units * self.n_nodes

    @property
    def output_size(self):
        return self._num_units * self.n_nodes

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        feat_out = self._num_units
        inputs_shape = inputs.get_shape()
        feat_in = int(int(inputs_shape[1]) / self.n_nodes)

        scope = vs.get_variable_scope()
        with vs.variable_scope(scope):
            W1 = tf.get_variable("weights1", [
                self.K * (feat_in + feat_out), feat_out * 2], dtype=tf.float32, initializer=self._kernel_initializer)
            W2 = tf.get_variable("weights2", [
                self.K * (feat_in + feat_out), feat_out], dtype=tf.float32, initializer=self._kernel_initializer)

            bias1 = tf.get_variable("biases1", [
                feat_out * 2], dtype=tf.float32, initializer=self._bias_initializer)
            bias2 = tf.get_variable("biases2", [
                feat_out], dtype=tf.float32, initializer=self._bias_initializer)

        concat = tf.concat([inputs, state], 1)
        value = chebnet(
            concat, self.conv_matrix, W1, self.K, feat_out, self.n_nodes, 2)
        value = tf.reshape(value, [-1, self.n_nodes, feat_out * 2])
        value = tf.nn.bias_add(value, bias1)
        value = tf.reshape(value, [-1, self.n_nodes * feat_out * 2])
        value = sigmoid(value)

        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * state
        concat = tf.concat([inputs, r_state], 1)
        c = chebnet(concat, self.conv_matrix, W2,
                    self.K, feat_out, self.n_nodes, 1)
        c = tf.reshape(c, [-1, self.n_nodes, feat_out])
        c = tf.nn.bias_add(c, bias2)
        c = tf.reshape(c, [-1, self.n_nodes * feat_out])
        c = self._activation(c)

        new_h = u * state + (1 - u) * c
        return new_h, new_h