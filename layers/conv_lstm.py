from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import rnn_cell_impl
import tensorflow as tf
from layers.graph_convolution import gcn

class GConvLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units,
                 forget_bias=1.0,
                 activation=None, reuse=None, conv_matrix=None, use_residual_connection=False, projection_fn=None,
                 kernel_initializer=None, bias_initializer=tf.zeros_initializer()):
        super(GConvLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh
        self.conv_matrix = conv_matrix
        self.n_nodes = int(conv_matrix.get_shape()[0])
        self.use_residual_connection = use_residual_connection
        self.projection_fn = projection_fn
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return rnn_cell_impl.LSTMStateTuple(self._num_units * self.n_nodes, self._num_units * self.n_nodes)

    @property
    def output_size(self):
        return self._num_units * self.n_nodes

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h = state
        feat_out = self._num_units
        inputs_shape = inputs.get_shape()
        feat_in = int(inputs_shape[1]) // self.n_nodes

        scope = vs.get_variable_scope()
        with vs.variable_scope(scope):
            Wx = tf.get_variable("input_weights", [
                feat_in, feat_out], dtype=tf.float32, initializer=self._kernel_initializer)
            Wh = tf.get_variable("hidden_weights", [
                feat_out, feat_out], dtype=tf.float32, initializer=self._kernel_initializer)
            W = tf.get_variable("mutual_weights", [
                                2 * feat_out, feat_out * 3], dtype=tf.float32, initializer=self._kernel_initializer)
            bias = tf.get_variable("biases", [
                feat_out * 3], dtype=tf.float32, initializer=self._bias_initializer)
        conv_inputs = tf.nn.relu(
            gcn(inputs, self.conv_matrix, Wx, None, feat_in, feat_out, self.n_nodes))
        conv_hidden = tf.nn.relu(
            gcn(h, self.conv_matrix, Wh, None, feat_out, feat_out, self.n_nodes))
        concat = array_ops.concat([conv_inputs, conv_hidden], 1)
        value = gcn(
            concat, self.conv_matrix, W, bias, 2*feat_out, 3 * feat_out, self.n_nodes)

        value = tf.reshape(value, [-1, self.n_nodes, feat_out * 3])
        value = tf.nn.bias_add(value, bias)
        value = tf.reshape(value, [-1, self.n_nodes * feat_out * 3])

        i, j, o = array_ops.split(value=value, num_or_size_splits=3, axis=1)
        # tied gate
        i = sigmoid(i)
        new_c = (1 - i) * c + i * self._activation(j)
        output_gate = sigmoid(o)
        new_h = self._activation(new_c) * output_gate
        new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)

        if self.use_residual_connection:
            if new_h.get_shape().as_list() != inputs.get_shape().as_list():
                return new_h + self.projection_fn(inputs) * output_gate, new_state
            else:
                return new_h + inputs * output_gate, new_state
        else:
            return new_h, new_state

