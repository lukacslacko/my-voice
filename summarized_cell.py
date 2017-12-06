import tensorflow as tf

class SummarizedCell:
    def __init__(self, 
                 signal_window_size, predicted_summary_window_size,
                 summary_layers, prediction_layers):
        self.signal_window_size = signal_window_size
        self.predicted_summary_window_size = predicted_summary_window_size

        self.summary_layers = summary_layers
        self.prediction_layers = prediction_layers

        self.summary_W = []
        self.summary_b = []
        prev_dim = self.signal_window_size
        for curr_dim in summary_layers:
            self.summary_W += [tf.Variable(tf.random_uniform([prev_dim, curr_dim], minval=-1))]
            self.summary_b += [tf.Variable(tf.random_uniform([curr_dim], minval=-1))]

        self.prediction_W = []
        self.prediction_b = []
        prev_dim = self.signal_window_size + self.predicted_summary_window_size
        for curr_dim in prediction_layers:
            self.prediction_W += [tf.Variable(tf.random_uniform([prev_dim, curr_dim], minval=-1))]
            self.prediction_b += [tf.Variable(tf.random_uniform([curr_dim], minval=-1))]

    def ComputeSummary(self, signal_tensor):
        result = signal_tensor
        for i in range(len(self.summary_layers)):
            result = tf.matmul(result, self.summary_W[i]) + self.summary_b[i]
            if i < len(self.summary_layers) - 1:
                result = tf.sigmoid(result)
        return result

    def ComputePrediction(self, signal_tensor, predicted_summary_tensor):
        result = tf.concat([signal_tensor, predicted_summary_tensor], 1)
        for i in range(len(self.prediction_layers)):
            result = tf.matmul(result, self.prediction_W[i]) + self.prediction_b[i]
            if i < len(self.summary_layers) - 1:
                result = tf.sigmoid(result)
        return result
