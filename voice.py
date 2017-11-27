#!/usr/bin/python3

import scipy
import scipy.io
import scipy.io.wavfile

import tensorflow as tf

import math

def load_voice():
    wav = scipy.io.wavfile.read(R"C:\Users\lukac\OneDrive\out2.wav")
    return [wav[1][i][0] for i in range(int(wav[1].size/2))]

v = load_voice()
print(min(v), max(v))
