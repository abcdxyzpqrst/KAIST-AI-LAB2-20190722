import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras import layers
from tensorflow.python.layers.core import Dense

class ConcreteDropout(layers.Wrapper):
    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))
        
    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # initialize p
        self.p_logit = self.add_variable(name='p_logit',
                                         shape=None,
                                         initializer=tf.random_uniform(
                                             (1,),
                                             self.init_min,
                                             self.init_max),
                                         dtype=tf.float32,
                                         trainable=True)
        self.p = tf.nn.sigmoid(self.p_logit[0])
        tf.add_to_collection("LAYER_P", self.p)

        # initialize regularizer / prior KL term
        input_dim = int(np.prod(input_shape[1:]))

        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
            weight)) / (1. - self.p)

def main():
    ConcreteDropout()
    return

if __name__ == "__main__":
    main()
