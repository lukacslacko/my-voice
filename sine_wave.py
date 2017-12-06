import tensorflow as tf
import numpy as np
import random
import math
import cell

class SineWave:
    def __init__(self):
        pass

    def generate_training_data(self, size, input_window_size, prediction_window_size):
        xs = []
        ys = []
        for _ in range(size):
            x = []
            y = []
            a = random.random()
            p = 2 * math.pi * random.random()
            d = 0.05 + 0.15 * random.random() 
            for _ in range(input_window_size):
                x = x + [a * math.sin(p)]
                p = p + d
            for _ in range(prediction_window_size):
                y = y + [a * math.sin(p)]
                p = p + d
            xs = xs + [x]
            ys = ys + [y]
        return xs, ys
