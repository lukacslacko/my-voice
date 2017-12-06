#!/usr/bin/python3

import random
import scipy
import scipy.io
import scipy.io.wavfile

class Voice:
    def __init__(self):
        wav = scipy.io.wavfile.read(R"C:\Users\lukac\OneDrive\out2.wav")
        data = [wav[1][i][0] for i in range(int(wav[1].size/2))]
        amp = max(abs(min(data)), abs(max(data)))/10
        self.signal = [d/amp for d in data]

    def generate_training_data(self, size, input_window_size, prediction_window_size):
        xs = []
        ys = []
        length = input_window_size + prediction_window_size
        for _ in range(size):
            offset = random.randint(0, len(self.signal) - length - 1)
            x = []
            y = []
            for _ in range(input_window_size):
                x = x + [self.signal[offset]]
                offset += 1
            for _ in range(prediction_window_size):
                y = y + [self.signal[offset]]
                offset += 1
            xs = xs + [x]
            ys = ys + [y]
        return xs, ys