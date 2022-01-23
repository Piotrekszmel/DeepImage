import tensorflow as tf

def net(image):
    conv1 = Conv2D(image, 32, 9, 1)
    conv2 = Conv2D(conv1, 64, 3, 2)
    conv3 = Conv2D(conv2, 128, 3, 2)
    resid1 = ResidualBlock(conv3, 3)
    resid2 = ResidualBlock(resid1, 3)
    resid3 = ResidualBlock(resid2, 3)
    resid4 = ResidualBlock(resid3, 3)
    resid5 = ResidualBlock(resid4, 3)
    conv_t1 = Conv2DTranspose(resid5, 64, 3, 2)
    conv_t2 = Conv2DTranspose(conv_t1, 32, 3, 2)
    conv_t3 = Conv2D(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def Conv2D(data, num_filters: int, filter_size: int, strides: int, activation: bool =True):
    weights = _conv_init_weights(data, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    output = tf.nn.conv2d(data=data, filters=weights, strides=strides_shape, padding='SAME')
    output = _normalize(output)
    if activation:
        return tf.nn.relu(output)
    return output

def Conv2DTranspose(data, num_filters: int, filter_size: int, strides: int):
    weights = _conv_init_weights(data, num_filters, filter_size, transpose=True)

    batch_size, rows, cols = data.get_shape()[:3]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides,1]
    
    output = tf.nn.conv2d_transpose(data, weights, tf_shape, strides_shape, padding='SAME')
    output = _normalize(output)
    return tf.nn.relu(output)

def ResidualBlock(data, filter_size: int = 3):
    conv2d = Conv2D(data, 128, filter_size, 1)
    return data + Conv2D(conv2d, 128, filter_size, 1, relu=False)

def _normalize(data):
    channels = data.get_shape[-1]
    mu, sigma_sq = tf.nn.moments(x=data, axes=[1,2], keepdims=True)
    shift = tf.Variable(tf.zeros([channels]))
    scale = tf.Variable(tf.ones([channels]))
    epsilon = 1e-3
    normalized = (data - mu) / (sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _conv_init_weights(net, out_channels: int, filter_size: int, transpose: bool = False):
    in_channels = net.get_shape()[-1]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    return tf.Variable(
        tf.random.truncated_normal(weights_shape,
                                   stddev=.1,
                                   seed=1),
        dtype=tf.float32)
    