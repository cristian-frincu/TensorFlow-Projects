import math
import tensorflow as tf
import numpy as np

HIDDEN_NODES = 3

x = tf.placeholder(tf.float32,[None,2])
W_hidden = tf.Variable(tf.truncated_normal([2,HIDDEN_NODES], stddev=1./math.sqrt(2)))
b_hidden = tf.Variable(tf.zeros([HIDDEN_NODES]))
hidden = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

W_logits =  tf.Variable(tf.truncated_normal([HIDDEN_NODES, 2], stddev=1./math.sqrt(HIDDEN_NODES)))
b_logits = tf.Variable(tf.zeros([2]))
logits = tf.matmul(hidden, W_logits)+b_logits

y = tf.nn.softmax(logits)

y_input = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_input)

loss = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

xTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yTrain = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])


for i in xrange(200):
    loss_val = sess.run([train_op, loss], feed_dict={x: xTrain, y_input: yTrain})
    if i% 10 == 0:
        print "Step:", i, "Current loss:", loss_val
        for x_input in [[0,0], [0, 1], [1, 0], [1, 1]]:
            print x_input, sess.run(y, feed_dict={x: [x_input]})

print "--Hidden--"
print W_hidden.eval(sess)
print b_hidden.eval(sess)
print "--Logits--"
print W_logits.eval(sess)
print b_logits.eval(sess)



