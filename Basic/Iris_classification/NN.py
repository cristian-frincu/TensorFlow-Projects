import math
import tensorflow as tf
import numpy as np
import csv
import sys

HIDDEN_NODES = 10
INPUT_NODES = 4
OUTPUT_NODES = 1

x = tf.placeholder(tf.float32,[None,INPUT_NODES])

W_hidden = tf.Variable(tf.truncated_normal([INPUT_NODES,HIDDEN_NODES], stddev=1./math.sqrt(HIDDEN_NODES)), name="W_hidden")
b_hidden = tf.Variable(tf.zeros([HIDDEN_NODES]), name="b_hidden")
hidden = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

W_logits =  tf.Variable(tf.truncated_normal([HIDDEN_NODES, OUTPUT_NODES], stddev=1./math.sqrt(HIDDEN_NODES)), name= "W_logits")
b_logits = tf.Variable(tf.zeros([OUTPUT_NODES]), name="b_logits")
logits = tf.matmul(hidden, W_logits) + b_logits

y = tf.nn.softmax(logits)

y_input = tf.placeholder(tf.float32, [None, OUTPUT_NODES])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_input)

loss = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)


xTrain = []
yTrain = []





def irisNameToList(name):
    if name == "Iris-setosa":
        return [1, 0, 0]
    elif name == "Iris-versicolor":
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def irisNameToFloat(name):
    if name == "Iris-setosa":
        return -1.
    elif name == "Iris-versicolor":
        return 0.
    else:
        return 1.

with open('iris.data', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)


for x in range(len(your_list[0:])-1):
    xTrain.append([float(your_list[x][0]), float(your_list[x][1]), float(your_list[x][2]), float(your_list[x][3])])
    # yTrain.append(irisNameToList(your_list[x][4]))
    yTrain.append([irisNameToFloat(your_list[x][4])])
    # print irisNameToFloat(your_list[x][4])
xTrain=[[1,1,2,1]]
yTrain = [[1]]


saver = tf.train.Saver([W_hidden, b_hidden, W_logits, b_logits])

for i in xrange(400):
    loss_val = sess.run([train_op, loss], feed_dict={x: xTrain, y_input: yTrain})
    if i% 10 == 0:
        print "Step:", i, "Current loss:", loss_val

    # if i % 100 == 0:
    #     saver.save(sess,'Checkpoint3', global_step=i)

#
# print "--Hidden--"
# print W_hidden.eval(sess)
# print b_hidden.eval(sess)
# print "--Logits--"
# print W_logits.eval(sess)
# print b_logits.eval(sess)


#done with the training, now testing

xTest = []
yTest = []

with open('iris.data', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

for sLength,sWidth, pLength,pWidth, irisName in your_list[0]:
    xTest.append([sLength,sWidth,pLength,pWidth])
    yTest.append(irisNameToList(irisName))



for x_test in xTrain:
    nnResult = sess.run(y, feed_dict={x: [x_test]})
    print nnResult



