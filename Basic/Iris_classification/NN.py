import math
import tensorflow as tf
import numpy as np
import csv
import sys

HIDDEN_NODES = 750
INPUT_NODES = 4
OUTPUT_NODES = 3

x = tf.placeholder(tf.float32,[None,INPUT_NODES])
W_hidden = tf.Variable(tf.truncated_normal([INPUT_NODES,HIDDEN_NODES], stddev=1./math.sqrt(HIDDEN_NODES)), name="W_hidden")
b_hidden = tf.Variable(tf.zeros([HIDDEN_NODES]), name="b_hidden")
hidden = tf.nn.relu6(tf.matmul(x, W_hidden) + b_hidden)

W_logits =  tf.Variable(tf.truncated_normal([HIDDEN_NODES, OUTPUT_NODES], stddev=1./math.sqrt(HIDDEN_NODES)), name= "W_logits")
b_logits = tf.Variable(tf.zeros([OUTPUT_NODES]), name="b_logits")
logits = tf.matmul(hidden, W_logits)+b_logits

y = tf.nn.softmax(logits)

y_input = tf.placeholder(tf.float32, [None, OUTPUT_NODES])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_input)

loss = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)


xTrain = []
yTrain = []


with open('iris.data', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

def irisNameToList(name):
    if name == "Iris-setosa":
        return [1, 0, 0]
    elif name == "Iris-versicolor":
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def irisNameToFloat(name):
    if name == "Iris-setosa":
        return -1
    elif name == "Iris-versicolor":
        return 0
    else:
        return 1
#initial restuls are 1 and -1, mapping 1 ->1 , -1 -> 0


for i in your_list:
    xTrain.append([i[0],i[1],i[2],i[3]])
    yTrain.append(irisNameToList(i[4]))



# print xTrain

# xTrain=[[1.2,2.4,2.1,1.6]]
# yTrain = [[1,1]]


saver = tf.train.Saver([W_hidden, b_hidden, W_logits, b_logits])

for i in xrange(600):
    loss_val = sess.run([train_op, loss], feed_dict={x: xTrain, y_input: yTrain})
    if i% 10 == 0:
        print "Step:", i, "Current loss:", loss_val
        if loss_val[1] < 0.0:
            print "neg"
            break
        if loss_val[1] > 4:
            print "too big",loss_val[1]
            sys.exit()

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


with open('bezdekIris.data', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

for i in your_list:
    xTest.append([i[0],i[1],i[2],i[3]])
    yTest.append(irisNameToList(i[4]))


count =0
goodResults=0

def roundListResults(my_list):
    max_value = max(my_list[0])
    max_index = list(my_list[0]).index(max_value)
    if max_index == 0 :
        return [1, 0, 0]
    if max_index == 1:
        return [0, 1, 0]
    else:
        return [0, 0, 1]


for x_test in xTest:
    nnResult = sess.run(y, feed_dict={x: [x_test]})

    if roundListResults(nnResult) == yTest[count]:
        goodResults += 1
    else:
        print nnResult,
        print roundListResults(nnResult),  yTest[count]
    count += 1

print "Correct predictions: ", goodResults
print "Total tests: ", count
print "Percentage Correct", float(goodResults)/count
# print "Average Pos value: ", float(sumPosValues)/posCount
# print "Average Neg value: ", float(sumNegValues)/negCount
# print "Number of pos", posCount
# print "Number of neg", negCount



