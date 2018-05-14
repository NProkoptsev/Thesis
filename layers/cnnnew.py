import tensorflow as tf
from layers.graph_convolution import GCN, gcn, gcn2


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]


def temporal_convolution_layer(inputs, output_units, convolution_width, causal=False, dilation_rate=[1], bias=True,
                               activation=None, dropout=None, scope='temporal-convolution-layer', reuse=False):
    """
    Convolution over the temporal axis of sequence data.
    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of timesteps to use in convolution.
        causal: Output at timestep t is a function of inputs at or before timestep t.
        dilation_rate:  Dilation rate along temporal axis.
    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        init_shape = shape(inputs)
        inputs = tf.transpose(inputs, [0, 1, 3, 2])
        if causal:
            shift = (convolution_width // 2) + (int(dilation_rate[0] - 1))
            pad = tf.zeros([tf.shape(inputs)[0], shift,
                            shape(inputs, 2), shape(inputs, 3)])
            inputs = tf.concat([pad, inputs], axis=1)

        W = tf.get_variable(
            name='weights',
            initializer=tf.random_normal_initializer(
                mean=0,
                stddev=1.0 / tf.sqrt(float(convolution_width)
                                     * float(shape(inputs, 2)))
            ),
            shape=[convolution_width, shape(
                inputs, 2), shape(inputs, 3), output_units]
        )
        z = tf.nn.depthwise_conv2d(
            inputs, W, [1, 1, 1, 1], 'VALID', rate=[dilation_rate[0], 1])
        
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        z = tf.reshape(
            z, [-1, init_shape[1], init_shape[2],  output_units])
            
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        return z


def time_distributed_dense_layer(inputs, output_units, conv_matrix, bias=True, activation=None, batch_norm=None,
                                 dropout=None, scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].
    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.
    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.random_normal_initializer(
                mean=0.0, stddev=1.0 / float(shape(inputs, -1))),
            shape=[shape(inputs, -1), output_units]
        )
        b = None
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
        z = tf.reshape(inputs, [-1,shape(inputs,2), shape(inputs,3)])

        z = gcn(z, conv_matrix, W,
                b, shape(inputs, -1), output_units, shape(inputs, -2))

        z = tf.reshape(z, [-1, shape(inputs,1), shape(inputs,2), output_units])
        if batch_norm is not None:
            z = tf.layers.batch_normalization(
                z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


class WaveNet():
    def __init__(
        self,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(6)]*2,
        filter_widths=[2 for i in range(6)]*2,
        num_decode_steps=12,
        conv_matrix=None,
        params = None,
        **kwargs
    ):
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.filter_widths = filter_widths
        self.num_decode_steps = num_decode_steps
        self.conv_matrix = conv_matrix
        self.enc_len = 64
        self.dec_len = 12
        self.params = params
    

    def encode(self, features):
        inputs = time_distributed_dense_layer(
            inputs=features,
            output_units=self.residual_channels,
            conv_matrix = self.conv_matrix,
            activation=tf.nn.tanh,
            scope='x-proj-encode'
        )
        skip_outputs = []
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv = temporal_convolution_layer(
                inputs=inputs,
                output_units=2*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-encode-{}'.format(i)
            )

            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=3)
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                conv_matrix = self.conv_matrix,
                scope='dilated-conv-proj-encode-{}'.format(i)
            )
            skips, residuals = tf.split(
                outputs, [self.skip_channels, self.residual_channels], axis=3)
            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=3))
        h = time_distributed_dense_layer(
            skip_outputs, 128, conv_matrix = self.conv_matrix,scope='dense-encode-1', activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 1, conv_matrix = self.conv_matrix, scope='dense-encode-2')[:,-1]

        batch_size = tf.shape(features)[0]
        self.encode_len = tf.tile([self.enc_len], [batch_size])
        self.decode_len = tf.tile([self.dec_len], [batch_size])
        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(conv_inputs, self.dilations)):
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            batch_idx = tf.reshape(batch_idx, [-1])

            queue_begin_time = self.encode_len - dilation - 1
            temporal_idx = tf.expand_dims(
                queue_begin_time, 1) + tf.expand_dims(tf.range(dilation), 0)
            
            temporal_idx = tf.reshape(temporal_idx, [-1])

            idx = tf.stack([batch_idx, temporal_idx], axis=1)
            slices = tf.reshape(tf.gather_nd(conv_input, idx),
                                (batch_size, dilation, shape(conv_input, 2), shape(conv_input, 3)))

            layer_ta = tf.TensorArray(
                dtype=tf.float32, size=dilation + self.num_decode_steps)
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2, 3)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        # initialize output tensor array
        emit_ta = []
        emit_ta.append(y_hat)

        # initialize other loop vars
        for i in range(0,11):
            current_input = y_hat
            with tf.variable_scope('x-proj-encode', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                x_proj = tf.nn.tanh(gcn2(current_input, self.conv_matrix, w_x_proj,
                    b_x_proj, shape(current_input, -1), shape(w_x_proj,-1), shape(current_input, -2)))
            skip_outputs, updated_queues = [], []
            for i, (conv_input, queue, dilation) in enumerate(zip(conv_inputs, state_queues, self.dilations)):

                state = queue.read(i)
                with tf.variable_scope('dilated-conv-encode-{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights')#.format(i))
                    b_conv = tf.get_variable('biases')#.format(i))
                    dilated_conv = tf.einsum('ijk,kjm->ijm', state,w_conv[0]) + tf.einsum(
                        'ijk,kjm->ijm', x_proj,w_conv[1]) + b_conv

                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
                dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

                with tf.variable_scope('dilated-conv-proj-encode-{}'.format(i), reuse=True):
                    w_proj = tf.get_variable('weights')#.format(i))
                    b_proj = tf.get_variable('biases')#.format(i))
                    concat_outputs = gcn2(dilated_conv, self.conv_matrix, w_proj,
                        b_proj, shape(dilated_conv, -1), shape(w_proj,-1), shape(dilated_conv, -2))
                    
                skips, residuals = tf.split(
                    concat_outputs, [self.skip_channels, self.residual_channels], axis=2)

                x_proj += residuals
                skip_outputs.append(skips)
                queue.write(i + dilation, x_proj)

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
            with tf.variable_scope('dense-encode-1', reuse=True):
                w_h = tf.get_variable('weights')
                b_h = tf.get_variable('biases')
                h = tf.nn.relu(gcn2(skip_outputs, self.conv_matrix, w_h,
                        b_h, shape(skip_outputs, -1), shape(w_h,-1), shape(skip_outputs, -2)))

            with tf.variable_scope('dense-encode-2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                y_hat = gcn2(h, self.conv_matrix, w_y,
                        b_y, shape(h, -1), shape(w_y, -1), shape(h, -2))
            
            emit_ta.append(y_hat)
        outputs = tf.stack(emit_ta, axis = 1)
        return outputs

    def build(self, features, labels, mode, params):
        outputs = self.encode(
            features)
        return outputs
