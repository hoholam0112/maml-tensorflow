import tensorflow as tf

init_kernel = tf.contrib.layers.xavier_initializer()

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, th=0.1):
    return tf.maximum(th * x, x)

def batch_norm(x, is_training_pl):
    return tf.layers.batch_normalization(x, training=is_training_pl)

def dense(x, num_hidden_units, bias=True):
    return tf.layers.dense(x, num_hidden_units, kernel_initializer=init_kernel, use_bias = bias)

def trans_conv2d(x, num_filters, filter_size, st, pad='same', bias=True):
    return tf.layers.conv2d_transpose(x, num_filters, filter_size, strides = st, padding = pad, kernel_initializer=init_kernel, use_bias = bias)

def conv2d(x, num_filters, filter_size, st, pad='same', bias=True):
    return tf.layers.conv2d(x, num_filters, filter_size, strides = st, padding = pad, kernel_initializer=init_kernel, use_bias = bias)

def drop_out(x, rate, is_training_pl):
    return tf.layers.dropout(x, rate=rate, training=is_training_pl)

def max_pool(x, k, s, pad = 'SAME'):
    return tf.nn.max_pool(x, ksize= [1,k,k,1], strides = [1,s,s,1], padding= pad)

class OmniglotModel():
    def __init__(self, n_way):
        """
            n_way: the number of classes for task
        """
        self.n_way = n_way
        self.input_shape = [None, 28, 28, 1]

    def classifier(self, inp, is_training, reuse=False):
        with tf.variable_scope('classifier', reuse=reuse):
            h1 = relu(batch_norm(conv2d(inp, 64, 3, 2), is_training))
            h2 = relu(batch_norm(conv2d(h1, 64, 3, 2), is_training))
            h3 = relu(batch_norm(conv2d(h2, 64, 3, 2), is_training))
            h4 = relu(batch_norm(conv2d(h3, 64, 3, 2), is_training))
            pred = tf.nn.softmax(dense(tf.layers.flatten(h4), self.n_way))

        return pred


class MiniImageNet():
    def __init__(self, n_way):
        """
            n_way: the number of classes for task
        """
        self.n_way = n_way
        self.input_shape = [None, 84, 84, 3]

    def classifier(self, inp, is_training, reuse=False):
        with tf.variable_scope('classifier', reuse=reuse):
            h1 = max_pool(relu(batch_norm(conv2d(inp, 32, 3, 1), is_training)), 2, 2)
            h2 = max_pool(relu(batch_norm(conv2d(h1, 32, 3, 1), is_training)), 2, 2)
            h3 = max_pool(relu(batch_norm(conv2d(h2, 32, 3, 1), is_training)), 2, 2)
            h4 = max_pool(relu(batch_norm(conv2d(h3, 32, 3, 1), is_training)), 2, 2)
            pred = tf.nn.softmax(dense(tf.layers.flatten(h4), self.n_way))

        return pred

        









