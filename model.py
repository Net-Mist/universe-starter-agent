import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None, 42, 42, 1], name='X')

        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])

        size = 256
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


class VINPolicy(object):
    def __init__(self, ob_space, ac_space):
        # self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.x = tf.placeholder(tf.float32, [None, 4, 42, 42], name='X')
        x = tf.transpose(self.x, [0, 2, 3, 1], name='Xt')
        # print(1, ob_space)(42, 42, 1)
        # print(3, [None] + list(ob_space))[None, 42, 42, 1]

        # Compute the state
        with tf.variable_scope('State'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(3):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 32, 32],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
                    # has the ability to see the whole state

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 160],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [160], initializer=tf.constant_initializer(0))
                state = tf.matmul(hidden_state, w) + b
        self.state = state

        # Compute R
        with tf.variable_scope('Reward'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(3):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 32, 32],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
                    # has the ability to see the whole state

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 160],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [160], initializer=tf.constant_initializer(0))
                r = tf.matmul(hidden_state, w) + b
        self.r = r

        # VIN Part
        v = tf.fill(tf.shape(r), 0.0)
        v = tf.expand_dims(v, 2)
        r = tf.expand_dims(r, 2)
        filters = None
        q = None
        with tf.variable_scope('vi') as scope:
            for irec in range(30):
                with tf.name_scope('iter%d' % irec):
                    if irec == 1:
                        scope.reuse_variables()
                    # concatenate V with R
                    v_concat = tf.concat([v, r], 2)

                    filters = tf.get_variable('weights', [3, 2, ac_space],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                              dtype=tf.float32)
                    conv = tf.nn.conv1d(v_concat, filters, 1, padding='SAME')
                    biases = tf.get_variable('bias', [ac_space], initializer=tf.constant_initializer(0.0))
                    q = tf.nn.bias_add(conv, biases, name="Q")
                    # activation_summary(Q)
                    v = tf.reduce_max(q, reduction_indices=[2], keep_dims=True, name="V")

        splits = tf.split(filters, ac_space, axis=2)
        for i in range(ac_space):
            splits2 = tf.split(splits[i], 2, axis=1)
            for j in range(2):
                splits3 = tf.split(splits2[j], 3, axis=0)
                for k in range(3):
                    tf.summary.scalar("transition_function/action" + str(i) + "_" + str(j) + "_" + str(k),
                                      splits3[k][0, 0, 0])

        # attention part
        with tf.name_scope('attention'):
            Qa_img = tf.multiply(q, tf.tile(tf.expand_dims(state, 2), [1, 1, ac_space]), name='Qa_img')
            Qa = tf.reduce_sum(Qa_img, [1], name="Qa")

        # reactive policy (dense layer with softmax?)
        with tf.name_scope('softmax_linear'):
            w = tf.get_variable("w", [ac_space, ac_space],
                                initializer=normalized_columns_initializer(0.01))
            biases = tf.get_variable('b_policy', [ac_space], initializer=tf.constant_initializer(0.0))

            self.logits = tf.matmul(Qa, w) + biases
            softact = tf.nn.softmax(self.logits, name='softact')

        # Second attention part for the V_f computation
        with tf.name_scope('attention2'):
            self.vf = tf.multiply(tf.reduce_sum(v, [2]), state, name='Vf_img')
            self.vf = tf.reduce_sum(self.vf, [1], name="Vf")

        # logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        # self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        # self.state_out = [None, None]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        c_init = np.zeros((1, 1), np.float32)
        h_init = np.zeros((1, 1), np.float32)
        return [c_init, h_init]

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class VIN2DPolicy(object):
    def __init__(self, ob_space, ac_space):
        # self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.x = tf.placeholder(tf.float32, [None, 4, 42, 42], name='X')
        x = tf.transpose(self.x, [0, 2, 3, 1], name='Xt')

        batch_size = tf.shape(x)[0]

        # Compute the state
        with tf.variable_scope('State'):
            # Start with convolution
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(3):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 32, 32],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            # Then convolution transpose
            with tf.variable_scope('conv_transpose{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 32, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                # deconv_shape = tf.pack([batch_size, 6, 6, 32])
                conv = tf.nn.conv2d_transpose(hidden_state, filter, [batch_size, 6, 6, 32], [1, 2, 2, 1],
                                              padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            with tf.variable_scope('conv_transpose{}'.format(1)):
                filter = tf.get_variable('filter', [3, 3, 1, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                # deconv_shape = tf.pack([batch_size, 11, 11, 1])
                conv = tf.nn.conv2d_transpose(hidden_state, filter, [batch_size, 11, 11, 1], [1, 2, 2, 1],
                                              padding='SAME')
                bias = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0))
                state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))

        # Compute R
        with tf.variable_scope('Reward'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(3):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 32, 32],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
                    # has the ability to see the whole state
            # Then convolution transpose
            with tf.variable_scope('conv_transpose{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 32, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                                     dtype=tf.float32),
                                         dtype=tf.float32)
                # deconv_shape = tf.pack([batch_size, 6, 6, 32])
                conv = tf.nn.conv2d_transpose(hidden_state, filter, [batch_size, 6, 6, 32], [1, 2, 2, 1],
                                              padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            with tf.variable_scope('conv_transpose{}'.format(1)):
                filter = tf.get_variable('filter', [3, 3, 1, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                                     dtype=tf.float32),
                                         dtype=tf.float32)
                # deconv_shape = tf.pack([batch_size, 11, 11, 1])
                conv = tf.nn.conv2d_transpose(hidden_state, filter, [batch_size, 11, 11, 1], [1, 2, 2, 1],
                                              padding='SAME')
                bias = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0))
                r = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))

        # VIN Part
        v = tf.fill(tf.shape(r), 0.0)
        # v = tf.expand_dims(v, 3)
        # r = tf.expand_dims(r, 3)
        q = None
        with tf.variable_scope('vi') as scope:
            for irec in range(30):
                with tf.name_scope('iter%d' % irec):
                    if irec == 1:
                        scope.reuse_variables()
                    # concatenate V with R
                    v_concat = tf.concat([v, r], 3)

                    filters = tf.get_variable('weights', [3, 3, 2, ac_space],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                              dtype=tf.float32)
                    conv = tf.nn.conv2d(v_concat, filters, [1, 1, 1, 1], padding='SAME')
                    biases = tf.get_variable('bias', [ac_space], initializer=tf.constant_initializer(0.0))
                    q = tf.nn.bias_add(conv, biases, name="Q")
                    # activation_summary(Q)
                    v = tf.reduce_max(q, reduction_indices=[3], keep_dims=True,
                                      name="V")  # TODO : reduction_indices is deprecated, use axis instead

        # attention part
        with tf.name_scope('attention'):
            Qa_img = tf.multiply(q, tf.tile(state, [1, 1, 1, ac_space]), name='Qa_img')
            Qa = tf.reduce_sum(Qa_img, [1, 2], name="Qa")

        # reactive policy (dense layer with softmax?)
        with tf.name_scope('softmax_linear'):
            w = tf.get_variable("w", [ac_space, ac_space],
                                initializer=normalized_columns_initializer(0.01))
            biases = tf.get_variable('b_policy', [ac_space], initializer=tf.constant_initializer(0.0))

            self.logits = tf.matmul(Qa, w) + biases
            softact = tf.nn.softmax(self.logits, name='softact')

        # Second attention part for the V_f computation
        with tf.name_scope('attention2'):
            self.vf = tf.multiply(v, state, name='Vf_img')
            self.vf = tf.reduce_sum(self.vf, [1, 2, 3], name="Vf")

        # logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        # self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        # self.state_out = [None, None]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        c_init = np.zeros((1, 1), np.float32)
        h_init = np.zeros((1, 1), np.float32)
        return [c_init, h_init]

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class VINDeeperCNNPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = tf.placeholder(tf.float32, [None, 4, 42, 42], name='X')
        x = tf.transpose(self.x, [0, 2, 3, 1], name='Xt')

        # Compute the state
        with tf.variable_scope('State'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(3):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 32, 32],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            # Here, hidden_state : 3*3*32
            with tf.variable_scope('conv{}'.format(4)):
                filter = tf.get_variable('filter', [2, 2, 32, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 160],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [160], initializer=tf.constant_initializer(0))
                state = tf.matmul(hidden_state, w) + b

        # Compute R
        with tf.variable_scope('Reward'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(3):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 32, 32],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            # Here, hidden_state : 3*3*32
            with tf.variable_scope('conv{}'.format(4)):
                filter = tf.get_variable('filter', [2, 2, 32, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                                     dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 160],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [160], initializer=tf.constant_initializer(0))
                r = tf.matmul(hidden_state, w) + b

        # VIN Part
        v = tf.fill(tf.shape(r), 0.0)
        v = tf.expand_dims(v, 2)
        r = tf.expand_dims(r, 2)
        filters = None
        q = None
        with tf.variable_scope('vi') as scope:
            for irec in range(30):
                with tf.name_scope('iter%d' % irec):
                    if irec == 1:
                        scope.reuse_variables()
                    # concatenate V with R
                    v_concat = tf.concat([v, r], 2)

                    filters = tf.get_variable('weights', [3, 2, ac_space],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                              dtype=tf.float32)
                    conv = tf.nn.conv1d(v_concat, filters, 1, padding='SAME')
                    biases = tf.get_variable('bias', [ac_space], initializer=tf.constant_initializer(0.0))
                    q = tf.nn.bias_add(conv, biases, name="Q")
                    # activation_summary(Q)
                    v = tf.reduce_max(q, reduction_indices=[2], keep_dims=True,
                                      name="V")  # TODO : reduction_indices is deprecated, use axis instead

        splits = tf.split(filters, ac_space, axis=2)
        for i in range(ac_space):
            splits2 = tf.split(splits[i], 2, axis=1)
            for j in range(2):
                splits3 = tf.split(splits2[j], 3, axis=0)
                for k in range(3):
                    tf.summary.scalar("transition_function/action" + str(i) + "_" + str(j) + "_" + str(k),
                                      splits3[k][0, 0, 0])

        # attention part
        with tf.name_scope('attention'):
            Qa_img = tf.multiply(q, tf.tile(tf.expand_dims(state, 2), [1, 1, ac_space]), name='Qa_img')
            Qa = tf.reduce_sum(Qa_img, [1], name="Qa")

        # reactive policy (dense layer with softmax?)
        with tf.name_scope('softmax_linear'):
            w = tf.get_variable("w", [ac_space, ac_space],
                                initializer=normalized_columns_initializer(0.01))
            biases = tf.get_variable('b_policy', [ac_space], initializer=tf.constant_initializer(0.0))

            self.logits = tf.matmul(Qa, w) + biases
            softact = tf.nn.softmax(self.logits, name='softact')

        # Second attention part for the V_f computation
        with tf.name_scope('attention2'):
            self.vf = tf.multiply(tf.reduce_sum(v, [2]), state, name='Vf_img')
            self.vf = tf.reduce_sum(self.vf, [1], name="Vf")

        # logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        # self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        # self.state_out = [None, None]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        c_init = np.zeros((1, 1), np.float32)
        h_init = np.zeros((1, 1), np.float32)
        return [c_init, h_init]

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class FFPolicy(object):
    def __init__(self, ob_space, ac_space):
        # self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.x = tf.placeholder(tf.float32, [None, 4, 42, 42], name='X')
        x = tf.transpose(self.x, [0, 2, 3, 1], name='Xt')
        # print(1, ob_space)(42, 42, 1)
        # print(3, [None] + list(ob_space))[None, 42, 42, 1]

        # Compute the state
        with tf.variable_scope('State'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(3):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 32, 32],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
                    # has the ability to see the whole state

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 160],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [160], initializer=tf.constant_initializer(0))
                state = tf.matmul(hidden_state, w) + b
        with tf.variable_scope('Action'):
            self.logits = linear(state, ac_space, "action", normalized_columns_initializer(0.01))
        with tf.variable_scope('Value'):
            self.vf = tf.reshape(linear(state, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        c_init = np.zeros((1, 1), np.float32)
        h_init = np.zeros((1, 1), np.float32)
        return [c_init, h_init]

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class FFWiderPolicy(object):
    def __init__(self, ob_space, ac_space):
        # self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.x = tf.placeholder(tf.float32, [None, 4, 42, 42], name='X')
        x = tf.transpose(self.x, [0, 2, 3, 1], name='Xt')
        # print(1, ob_space)(42, 42, 1)
        # print(3, [None] + list(ob_space))[None, 42, 42, 1]

        # Compute the state
        with tf.variable_scope('State'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(3):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 64, 64],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
                    # has the ability to see the whole state

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 160],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [160], initializer=tf.constant_initializer(0))
                state = tf.matmul(hidden_state, w) + b
        with tf.variable_scope('Action'):
            self.logits = linear(state, ac_space, "action", normalized_columns_initializer(0.01))
        with tf.variable_scope('Value'):
            self.vf = tf.reshape(linear(state, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        c_init = np.zeros((1, 1), np.float32)
        h_init = np.zeros((1, 1), np.float32)
        return [c_init, h_init]

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class FFWiderAndDeeperPolicy(object):
    def __init__(self, ob_space, ac_space):
        # self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.x = tf.placeholder(tf.float32, [None, 4, 42, 42], name='X')
        x = tf.transpose(self.x, [0, 2, 3, 1], name='Xt')
        # print(1, ob_space)(42, 42, 1)
        # print(3, [None] + list(ob_space))[None, 42, 42, 1]

        # Compute the state
        with tf.variable_scope('State'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(9):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 64, 64],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
                    # has the ability to see the whole state

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 160],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [160], initializer=tf.constant_initializer(0))
                state = tf.matmul(hidden_state, w) + b
        with tf.variable_scope('Action'):
            self.logits = linear(state, ac_space, "action", normalized_columns_initializer(0.01))
        with tf.variable_scope('Value'):
            self.vf = tf.reshape(linear(state, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        c_init = np.zeros((1, 1), np.float32)
        h_init = np.zeros((1, 1), np.float32)
        return [c_init, h_init]

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class FFDeeperPolicy(object):
    def __init__(self, ob_space, ac_space):
        # self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.x = tf.placeholder(tf.float32, [None, 4, 42, 42], name='X')
        x = tf.transpose(self.x, [0, 2, 3, 1], name='Xt')
        # print(1, ob_space)(42, 42, 1)
        # print(3, [None] + list(ob_space))[None, 42, 42, 1]

        # Compute the state
        with tf.variable_scope('State'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [3, 3, 4, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
            for i in range(9):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    filter = tf.get_variable('filter', [3, 3, 32, 32],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                             dtype=tf.float32)
                    conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                    bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                    hidden_state = tf.nn.elu(tf.nn.bias_add(conv, bias, name="hidden"))
                    # has the ability to see the whole state

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 160],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [160], initializer=tf.constant_initializer(0))
                state = tf.matmul(hidden_state, w) + b
        with tf.variable_scope('Action'):
            self.logits = linear(state, ac_space, "action", normalized_columns_initializer(0.01))
        with tf.variable_scope('Value'):
            self.vf = tf.reshape(linear(state, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        c_init = np.zeros((1, 1), np.float32)
        h_init = np.zeros((1, 1), np.float32)
        return [c_init, h_init]

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class DeepMindPolicy(object):
    def __init__(self, ob_space, ac_space):
        # self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.x = tf.placeholder(tf.float32, [None, 4, 84, 84], name='X')
        x = tf.transpose(self.x, [0, 2, 3, 1], name='Xt')
        # print(1, ob_space)(42, 42, 1)
        # print(3, [None] + list(ob_space))[None, 42, 42, 1]

        # Compute the state
        with tf.variable_scope('State'):
            with tf.variable_scope('conv{}'.format(0)):
                filter = tf.get_variable('filter', [8, 8, 4, 16],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(x, filter, [1, 4, 4, 1], padding='SAME')
                bias = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.tanh(tf.nn.bias_add(conv, bias, name="hidden"))

            with tf.variable_scope('conv{}'.format(1)):
                filter = tf.get_variable('filter', [4, 4, 16, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32)
                conv = tf.nn.conv2d(hidden_state, filter, [1, 2, 2, 1], padding='SAME')
                bias = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
                hidden_state = tf.nn.tanh(tf.nn.bias_add(conv, bias, name="hidden"))
                # has the ability to see the whole state

            hidden_state = flatten(hidden_state)

            with tf.variable_scope('Linear_State'):
                w = tf.get_variable("w", [hidden_state.get_shape()[1], 256],
                                    initializer=normalized_columns_initializer(0.01))
                b = tf.get_variable("b", [256], initializer=tf.constant_initializer(0))
                state = tf.nn.relu(tf.matmul(hidden_state, w) + b)

        with tf.variable_scope('Action'):
            self.logits = linear(state, ac_space, "action", normalized_columns_initializer(0.01))
        with tf.variable_scope('Value'):
            self.vf = tf.reshape(linear(state, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        c_init = np.zeros((1, 1), np.float32)
        h_init = np.zeros((1, 1), np.float32)
        return [c_init, h_init]

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class VINLSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None, 42, 42, 1], name='X')

        # Prepare LSTM
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])

        # LSTM
        size = 256
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]
        # Initial values
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        # Current values
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, x, initial_state=state_in, sequence_length=step_size,
                                                     time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        x = tf.reshape(lstm_outputs, [-1, size])

        # Then compute state and reward
        self.state = tf.nn.elu(linear(x, 160, "state", normalized_columns_initializer(0.01)))
        self.reward = tf.nn.elu(linear(x, 160, "reward", normalized_columns_initializer(0.01)))

        # VIN Part
        value_tensor = tf.fill(tf.shape(self.reward), 0.0)
        value_tensor = tf.expand_dims(value_tensor, 2)
        reward = tf.expand_dims(self.reward, 2)
        filters = None
        q = None
        with tf.variable_scope('vi') as scope:
            for irec in range(30):
                with tf.name_scope('iter%d' % irec):
                    if irec == 1:
                        scope.reuse_variables()
                    # concatenate V with R
                    value_concat = tf.concat([value_tensor, reward], 2)

                    filters = tf.get_variable('weights', [3, 2, ac_space],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                              dtype=tf.float32)
                    conv = tf.nn.conv1d(value_concat, filters, 1, padding='SAME')
                    biases = tf.get_variable('bias', [ac_space], initializer=tf.constant_initializer(0.0))
                    q = tf.nn.bias_add(conv, biases, name="Q")
                    # activation_summary(Q)
                    v = tf.reduce_max(q, reduction_indices=[2], keep_dims=True, name="V")

        splits = tf.split(filters, ac_space, axis=2)
        for i in range(ac_space):
            splits2 = tf.split(splits[i], 2, axis=1)
            for j in range(2):
                splits3 = tf.split(splits2[j], 3, axis=0)
                for k in range(3):
                    tf.summary.scalar("transition_function/action" + str(i) + "_" + str(j) + "_" + str(k),
                                      splits3[k][0, 0, 0])

        # attention part
        with tf.name_scope('attention'):
            Qa_img = tf.multiply(q, tf.tile(tf.expand_dims(self.state, 2), [1, 1, ac_space]), name='Qa_img')
            Qa = tf.reduce_sum(Qa_img, [1], name="Qa")
        self.logits = Qa


        # Second attention part for the V_f computation
        with tf.name_scope('attention2'):
            self.vf = tf.multiply(tf.reduce_sum(v, [2]), self.state, name='Vf_img')
            self.vf = tf.reduce_sum(self.vf, [1], name="Vf")

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
