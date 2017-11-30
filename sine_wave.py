import tensorflow as tf
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import cell

input_window_size = 16
prediction_window_size = 4

cell = cell.Cell(
    input_window_size=input_window_size,
    prediction_window_size=prediction_window_size,
    layer_sizes=[8])

def generate_training_data(size):
    xs = []
    ys = []
    for _ in range(size):
        x = []
        y = []
        a = random.random()
        p = 2 * math.pi * random.random()
        d = 0.1 + 0.3 * random.random() 
        for _ in range(input_window_size):
            x = x + [a * math.sin(p)]
            p = p + d
        for _ in range(prediction_window_size):
            y = y + [a * math.sin(p)]
            p = p + d
        xs = xs + [x]
        ys = ys + [y]
    return xs, ys

x = tf.placeholder(tf.float32, [None, input_window_size])
expected = tf.placeholder(tf.float32, [None, prediction_window_size])

out_activation = cell.Apply(x)

loss = tf.nn.l2_loss(out_activation - expected)

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

tf.summary.scalar('loss', loss)

sess = tf.InteractiveSession()

merged = tf.summary.merge_all()

# To start tensorboard:
# AppData\Local\Programs\Python\Python36\python.exe -m tensorboard.main --logdir=C:\Users\lukac\train
train_writer = tf.summary.FileWriter(r"C:\Users\lukac\train", sess.graph)
test_writer = tf.summary.FileWriter(r"C:\Users\lukac\test")

tf.global_variables_initializer().run()

test_input, test_expected = generate_training_data(10000)

for i in range(500000):
    input_example, expected_example = generate_training_data(100)
    sess.run(train_step, feed_dict={x: input_example, expected: expected_example})
    #print(i, input_example, expected_example)
    if i % 100 == 99:
        print(i, 
              sess.run(loss, feed_dict={x: input_example, expected: expected_example}), 
              sess.run(loss, feed_dict={x: test_input, expected: test_expected}))
        train_writer.add_summary(sess.run(merged, feed_dict={x: input_example, expected: expected_example}), i)
        test_writer.add_summary(sess.run(merged, feed_dict={x: test_input, expected: test_expected}), i)

train_writer.close()
test_writer.close()

while True:
    n = 5
    a, b = generate_training_data(n)

    def extrapolate(xs, steps):
        result = xs
        for _ in range(steps):
            last_x = [result[-input_window_size:]]
            pred = sess.run(out_activation, feed_dict={x: last_x})
            result = result + pred[0].tolist()
        return result

    for i in range(n):
        plt.plot(extrapolate(a[i], 10))
    plt.show()
