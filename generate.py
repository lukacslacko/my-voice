import matplotlib.pyplot as plt
import source
import tensorflow as tf
import voice_predictor
import sounddevice as sd

predictor = voice_predictor.VoicePredictor()

source = source.Source()

n = 5
a, _ = source.generate_training_data(
    n, predictor.input_window_size, predictor.prediction_window_size)

saver = tf.train.Saver()

sess = tf.InteractiveSession()

saver.restore(sess, r"C:\Users\lukac\voicetf\ckpt")

def extrapolate(xs, steps):
    result = xs
    for _ in range(steps):
        last_x = [result[-predictor.input_window_size:]]
        pred = sess.run(predictor.out_activation, feed_dict={predictor.x: last_x})
        result = result + pred[0].tolist()
    return result

for i in range(n):
    sound = extrapolate(a[i], 2000)
    sd.play(sound, 48000)

for i in range(n):
    plt.plot(extrapolate(a[i], 20))
plt.show()
