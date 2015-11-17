
import tensorflow as tf
import numpy as np
from itertools import product


def init_weights(shape):
    return tf.Variable(tf.zeros(shape))

def model(X, w_h, w_h2 , w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions

    h2 = tf.nn.sigmoid(tf.matmul(h, w_h2))

    return tf.matmul(h2, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])
teX, teY = trX, trY

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 1])

w_h = init_weights([2, 4]) # create symbolic variables
w_h2 = init_weights([4, 4]) # create symbolic variables
w_o = init_weights([4, 1])

py_x = model(X, w_h, w_h2, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.25).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(500):
    sess.run(train_op, feed_dict={X: trX, Y: trY})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))

print w_h.eval(sess)
print w_h2.eval(sess)
print w_o.eval(sess)



# for A,B in product([0,1],[0,1]):
#   top = W00*A + W01*A + B0
#   bottom = W10*B + W11*B + B1
#   print "A:",A," B:",B
#   # print "Top",top," Bottom: ", bottom
#   print "Sum:",top+bottom