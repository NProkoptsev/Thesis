"""Module implementing the IndRNN cell"""

import tensorflow as tf
from layers.graph_convolution import gcn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import base as base_layer


class GConvIndRNNCell(rnn_cell_impl.LayerRNNCell):
    def __init__(self,
                 num_units,
                 conv_matrix,
                 n_nodes,
                 recurrent_min_abs=0,
                 recurrent_max_abs=None,
                 recurrent_kernel_initializer=None,
                 input_kernel_initializer=None,
                 activation=nn_ops.relu,
                 projection_fn =None,
                 reuse=None,
                 name=None):
        super(GConvIndRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 3-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._conv_matrix = conv_matrix
        self._n_nodes = n_nodes
        self._recurrent_min_abs = recurrent_min_abs
        self._recurrent_max_abs = recurrent_max_abs
        self._recurrent_kernel_initializer = recurrent_kernel_initializer
        self._input_kernel_initializer = input_kernel_initializer
        self._activation = activation
        self._projection_fn = projection_fn

    @property
    def state_size(self):
        return self._num_units * self._n_nodes

    @property
    def output_size(self):
        return self._num_units * self._n_nodes

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        self._input_depth = inputs_shape[1].value
        self._filters_num = self._input_depth // self._n_nodes
        self._output_depth = self._n_nodes * self._num_units

        self._conv_kernel = self.add_variable("conv_kernel", [
            self._filters_num, self._num_units], dtype=self.dtype,
            initializer=self._input_kernel_initializer)

        self._bias = self.add_variable(
            "bias",
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        if self._recurrent_kernel_initializer is None:
            self._recurrent_kernel_initializer = init_ops.random_uniform_initializer(
                minval=0.,
                maxval=1,
            )

        self._recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            shape=[self._num_units * self._n_nodes], initializer=self._recurrent_kernel_initializer)

        # Clip the absolute values of the recurrent weights to the specified minimum
        if self._recurrent_min_abs and self._recurrent_min_abs != 0:
            abs_kernel = math_ops.abs(self._recurrent_kernel)
            min_abs_kernel = math_ops.maximum(
                abs_kernel, self._recurrent_min_abs)
            self._recurrent_kernel = math_ops.multiply(
                math_ops.sign(self._recurrent_kernel),
                min_abs_kernel
            )

        # Clip the absolute values of the recurrent weights to the specified maximum
        self._recurrent_max_abs = self._recurrent_max_abs or 1.
        self._recurrent_kernel = clip_ops.clip_by_value(self._recurrent_kernel,
                                                        -self._recurrent_max_abs,
                                                        self._recurrent_max_abs)

        self.built = True

    def call(self, inputs, state):
        value = gcn(inputs, self._conv_matrix, self._conv_kernel,
                       self._bias, self._filters_num, self._num_units, self._n_nodes) 
            
        #state = tf.reshape(state, [-1, self._n_nodes, self._num_units])
        recurrent_update = math_ops.multiply(state, self._recurrent_kernel)

        value = math_ops.add(value, recurrent_update)
        output = self._activation(value)
        #normalized_output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
        #output = tf.reshape(value, [-1, self._n_nodes * self._num_units])
        if self._num_units == self._filters_num:
            return output + inputs, output
        else:
            return output + self._projection_fn(inputs), output
