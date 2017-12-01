import sine_wave
import tensorflow as tf
import voice_predictor

predictor = voice_predictor.VoicePredictor()
expected = tf.placeholder(tf.float32, [None, predictor.prediction_window_size])

def generate_test_data(size):
    x, exp = sine_wave.generate_training_data(
        size, predictor.input_window_size, predictor.prediction_window_size)
    return {predictor.x: x, expected: exp}

def generate_training_data(size):
    x, exp = sine_wave.generate_training_data(
        size, predictor.input_window_size, predictor.prediction_window_size)
    return {predictor.x: x, expected: exp}


test_dict = generate_test_data(10000)

loss = tf.nn.l2_loss(predictor.out_activation - expected)

train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

tf.summary.scalar('loss', loss)

sess = tf.InteractiveSession()

merged = tf.summary.merge_all()

# To start tensorboard:
# AppData\Local\Programs\Python\Python36\python.exe -m tensorboard.main --logdir=C:\Users\lukac\voicetf\train
train_writer = tf.summary.FileWriter(r"C:\Users\lukac\voicetf\train", sess.graph)
test_writer = tf.summary.FileWriter(r"C:\Users\lukac\voicetf\test")

tf.global_variables_initializer().run()

saver = tf.train.Saver()

for i in range(500_000):
    train_dict = generate_training_data(100)
    sess.run(train_step, feed_dict=train_dict)
    #print(i, input_example, expected_example)
    if i % 100 == 99:
        print(i, 
              sess.run(loss, feed_dict=train_dict), 
              sess.run(loss, feed_dict=test_dict))
        train_writer.add_summary(sess.run(merged, feed_dict=train_dict), i)
        test_writer.add_summary(sess.run(merged, feed_dict=test_dict), i)
        saver.save(sess, r"C:\Users\lukac\voicetf\ckpt")

train_writer.close()
test_writer.close()
