import cell
import tensorflow as tf

class VoicePredictor:
    def __init__(self):
        self.input_window_size = 16
        self.prediction_window_size = 4
        self.cell = cell.Cell(
            input_window_size=self.input_window_size,
            prediction_window_size=self.prediction_window_size,
            layer_sizes=[8])

        self.x = tf.placeholder(tf.float32, [None, self.input_window_size])
        
        self.out_activation = self.cell.Apply(self.x)
        