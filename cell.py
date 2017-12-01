import tensorflow as tf

class Cell:
    def __init__(self, input_window_size, prediction_window_size, layer_sizes):
        self.input_window_size = input_window_size
        self.prediction_window_size = prediction_window_size
        self.layer_sizes = layer_sizes

        self.W = []
        self.b = []

        self.first_W = tf.Variable(tf.random_uniform([input_window_size, layer_sizes[0]], minval=-1))
        self.first_b = tf.Variable(tf.random_uniform([layer_sizes[0]], minval=-1))

        for i in range(len(layer_sizes)):
            this_size = layer_sizes[i]
            next_size = prediction_window_size if i == len(layer_sizes) - 1 else layer_sizes[i+1]
            self.W = self.W + [tf.Variable(tf.random_uniform([this_size, next_size], minval=-1))]
            self.b = self.b + [tf.Variable(tf.random_uniform([next_size], minval=-1))]


    def Apply(self, input_tensor):
        value = tf.matmul(input_tensor, self.first_W) + self.first_b
        for i in range(len(self.W)):
            value = tf.matmul(tf.sigmoid(value), self.W[i]) + self.b[i]
        return value
