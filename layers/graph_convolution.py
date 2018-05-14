import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.framework import tensor_shape

from tensorflow.python.ops import init_ops


def gcn(x, L, W, bias, feat_in, feat_out, n_nodes):
    '''
    x : [batch_size, n_nodes * feat_in] - input of each time step [32, 64, 100]
    batch_size : number of samples
    n_nodes : number of node in graph
    feat_in : number of input feature
    feat_out : number of output feature
    L : renormalized adj matrix [n_nodes, n_nodes]
    W : Weights [feat_in, feat_out*m]
    m : number of gates
    '''
    x = tf.reshape(x, [-1, n_nodes, feat_in])

    # cast to [n_nodes, feat_in, batch_size]
    x = tf.transpose(x, perm=[1, 2, 0])
    x = tf.reshape(x, [n_nodes, -1])  # cast to [n_nodes, feat_in * batch_size]

    # x = tf.sparse_tensor_dense_matmul(sp_a = L, b= x)  # n_nodes, feat_in*batch_size
    x = tf.matmul(L,x)

    x = tf.reshape(x, [n_nodes, feat_in, -1])
    x = tf.transpose(x, perm=[2, 0, 1])  # [batch_size, n_nodes, feat_in]
    x = tf.reshape(x, [-1, feat_in])    # cast to [batch_size*n_nodes, feat_in]

    x = tf.matmul(x, W)  # [batch_size * n_nodes, feat_out*m ]

    if bias:
        x = x + bias
    out = tf.reshape(x, [-1, n_nodes * feat_out])
    return out

def gcn2(x, L, W, bias, feat_in, feat_out, n_nodes):
    '''
    x : [batch_size, n_nodes * feat_in] - input of each time step [32, 64, 100]
    batch_size : number of samples
    n_nodes : number of node in graph
    feat_in : number of input feature
    feat_out : number of output feature
    L : renormalized adj matrix [n_nodes, n_nodes]
    W : Weights [feat_in, feat_out*m]
    m : number of gates
    '''
    x = tf.reshape(x, [-1, n_nodes, feat_in])

    # cast to [n_nodes, feat_in, batch_size]
    x = tf.transpose(x, perm=[1, 2, 0])
    x = tf.reshape(x, [n_nodes, -1])  # cast to [n_nodes, feat_in * batch_size]

    # x = tf.sparse_tensor_dense_matmul(sp_a = L, b= x)  # n_nodes, feat_in*batch_size
    x = tf.matmul(L, x)
    x = tf.reshape(x, [n_nodes, feat_in, -1])
    x = tf.transpose(x, perm=[2, 0, 1])  # [batch_size, n_nodes, feat_in]
    x = tf.reshape(x, [-1, feat_in])    # cast to [batch_size*n_nodes, feat_in]

    x = tf.matmul(x, W)  # [batch_size * n_nodes, feat_out*m ]

    if bias:
        x = x + bias
    out = tf.reshape(x, [-1, n_nodes, feat_out])
    return out

def chebnet(x, L, W, K, feat_out, n_nodes, m):
    '''
    x : [batch_size, n_nodes * feat_in] - input of each time step [32, 64, 100]
    batch_size : number of samples
    n_nodes : number of node in graph
    feat_in : number of input feature
    feat_out : number of output feature
    L : laplacian [K, n_nodes, n_nodes]
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out*m]
    m : number of simulataneous calculations
    '''
    _, feat_size = x.get_shape()
    feat_in = int(feat_size) // n_nodes
    x = tf.reshape(x, [-1, n_nodes, feat_in])

    # cast to [n_nodes, feat_in, batch_size]
    x = tf.transpose(x, perm=[1, 2, 0])
    x = tf.reshape(x, [n_nodes, -1])  # cast to [n_nodes, feat_in * batch_size]

    L = tf.reshape(L, [K * n_nodes, n_nodes])

    x = tf.matmul(L, x)  # K * n_nodes, feat_in*batch_size

    # cast to [K, n_nodes, feat_in, batch_size]
    x = tf.reshape(x, [K, n_nodes, feat_in, -1])
    x = tf.transpose(x, perm=[3, 1, 2, 0])
    # cast to [batch_size*n_nodes, feat_in*K]
    x = tf.reshape(x, [-1, feat_in * K])

    x = tf.matmul(x, W)

    out = tf.reshape(x, [-1, n_nodes * feat_out * m])
    return out


class GCN(base.Layer):
    def __init__(self, units, conv_matrix,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(GCN, self).__init__(trainable=trainable, name=name,
                                  **kwargs)
        self._num_units = units
        self._conv_matrix = conv_matrix
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `GCN` '
                             'should be defined. Found `None`.')
        self._input_depth = input_shape[-1].value
        self._n_nodes = input_shape[-2].value
        self._input_spec = base.InputSpec(min_ndim=2,
                                          axes={-1: input_shape[-1].value})

        self._conv_kernel = self.add_variable("conv_kernel", [
            self._input_depth, self._num_units],
            dtype=self.dtype,
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            trainable=True)

        if self._use_bias:
            self._bias = self.add_variable(
                "bias",
                shape=[self._num_units],
                initializer=self._bias_initializer,
                regularizer=self._bias_regularizer,
                trainable=True)
        else:
            self._bias = None
        self.built = True

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        outputs = gcn(inputs, self._conv_matrix, self._conv_kernel, self._bias, self._input_depth,
                      self._num_units, self._n_nodes)
        outputs = tf.reshape(outputs, [-1, self._n_nodes, self._num_units])
        if self._activation is not None:
            return self._activation(outputs)  # pylint: disable=not-callable
        return outputs

    # def compute_output_shape(self, input_shape):
    #     input_shape = tensor_shape.TensorShape(input_shape)
    #     input_shape = input_shape.with_rank_at_least(2)
    #     if input_shape[-1].value is None:
    #         raise ValueError(
    #             'The innermost dimension of input_shape must be defined, but saw: %s'
    #             % input_shape)
    #     return input_shape[:-1].concatenate(self.units)
