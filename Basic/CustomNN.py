import tensorflow as tf
import numpy as np

# The idea is to make a trivial XOR NN, on which I can make some tests, and see how I can convert the B and W to
#  structured text

filename_queue = tf.train.string_input_producer(["testInt.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1]]
col1, col2, col3= tf.decode_csv(
    value, record_defaults=record_defaults)
# print tf.shape(col1)


features = tf.pack([col1, col2, col3])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(4):
    # print i
    example, label = sess.run([features, col3])
    # print example, label

  coord.request_stop()
  coord.join(threads)




print "Done loading XOR table"

x = tf.placeholder("float", [None, 2])
W = tf.Variable(tf.random_normal([2, 2]))
b = tf.Variable(tf.random_normal([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 1])

print "Done init"

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

print "Done loading vars"

init = tf.initialize_all_variables()
print "Done: Initializing variables"

sess = tf.Session()
sess.run(init)
print "Done: Session started"

xTrain = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])
yTrain = np.array([[0], [1], [1], [0]])

for i in range(50):
  # print i
    sess.run(train_step, feed_dict={x: xTrain, y_: yTrain})

print b.eval(sess)
print W.eval(sess)

print "Done training"


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

xTest = np.array([[0, 0], [0, 1]])
yTest = np.array([[1], [1]])


print "Result:"
print sess.run(accuracy, feed_dict={x: xTest, y_: yTest})
