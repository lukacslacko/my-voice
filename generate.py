import matplotlib.pyplot as plt
import sine_wave
import tensorflow as tf
import voice_predictor

predictor = voice_predictor.VoicePredictor()

n = 5
a, _ = sine_wave.generate_training_data(
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
    plt.plot(extrapolate(a[i], 20))
plt.show()
