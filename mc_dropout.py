# Import packages for implementations
import tensorflow as tf

def net(x, dropout_rate=0.5):
    """
    This function build a LeNet-5-Caffe convolutional neural network
    (2 conv layers + 2 dense layers with structure 20-50-800-500)
    for MNIST dataset

    Args:
        x: the input data (MNIST)
        keep_prob: keeping probability for dropout layers in the network

    Returns:
        y: the predicted labels (0 ~ 9)
    """

    # MNIST data have 28 x 28 shape with one single channel (black/white)
    input_layer = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("mnist_model", reuse=reuse):
        # 1st convolutional layer
        with tf.variable_scope("conv1", reuse=reuse):
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=20,
                kernel_size=[5, 5],
                activation=tf.nn.relu
            )
            dropout1 = tf.layers.dropout(conv1, dropout_rate=dropout_rate, training=True)

        # 1st pooling layer
        pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=[2, 2])
        
        # 2nd convolutional layer
        with tf.variable_scope("conv2", reuse=reuse):
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=50,
                kernel_size=[5, 5],
                activation=tf.nn.relu
            )

        pool2 = tf.layers.max_pooling2d(inputs=conv2, dropout_rate=dropout_rate, training=True)
        
        # 1st dense layer
        with tf.variable_scope("dense1", reuse=reuse):
            pool2_flat = tf.reshape(pool2, [-1, 800])
            dense = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.relu)
            dropout3 = tf.layers.dropout(dense, dropout_rate=dropout_rate, training=True)

        # softmax layer
        with tf.variable_scope("dense2", reuse=reuse):
            logits = tf.layers.dense(inputs=dropout3, units=10)
            class_prob = tf.nn.softmax(logits, name="softmax_tensor")

    return logits, class_prob

def train():
    return

def main():
    net(None, 0.5)
    return

if __name__ == "__main__":
    main()
