import math
import tensorflow as tf
import numpy as np
import csv
import sys

HIDDEN_NODES = 1000

x = tf.placeholder(tf.float32,[None,8])
W_hidden = tf.Variable(tf.truncated_normal([8,HIDDEN_NODES], stddev=1./math.sqrt(HIDDEN_NODES)), name="W_hidden")
b_hidden = tf.Variable(tf.zeros([HIDDEN_NODES]), name="b_hidden")
hidden = tf.nn.relu6(tf.matmul(x, W_hidden) + b_hidden)

W_logits =  tf.Variable(tf.truncated_normal([HIDDEN_NODES, 2], stddev=1./math.sqrt(HIDDEN_NODES)), name= "W_logits")
b_logits = tf.Variable(tf.zeros([2]), name="b_logits")
logits = tf.matmul(hidden, W_logits)+b_logits

y = tf.nn.softmax(logits)

y_input = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_input)

loss = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.0000001).minimize(loss)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)


xTrain = []
yTrain = []


with open('schillingData.txt', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

# a function that returns the int representation of the protein as it wouldbe described in the sequence
def mapProteinToInt(input):
    legend = "ARNDCQEGHILKMFPSTWYV"
    result=[]

    for i in range(len(legend)):
        # print input[i]
        for j in range(len(input)):
            if legend[i]==input[j]:
                result.append(i)
    return result

#initial restuls are 1 and -1, mapping 1 ->1 , -1 -> 0

def mapProteinSequenceResult(input):
    if int(input) == -1:
        return 0
    else:
        return 1

for xInput,yInput in your_list:
    xTrain.append(mapProteinToInt(xInput))
    yTrain.append([int(mapProteinSequenceResult(yInput)), (-1*int(mapProteinSequenceResult(yInput)))])

print type(xTrain)
print type(yTrain)



saver = tf.train.Saver([W_hidden, b_hidden, W_logits, b_logits])

for i in xrange(400):
    loss_val = sess.run([train_op, loss], feed_dict={x: xTrain, y_input: yTrain})
    if i% 10 == 0:
        print "Step:", i, "Current loss:", loss_val
        if loss_val[1] < 0.0:
            print "neg"
            break
        if loss_val[1] > 0.1:
            print "too big",loss_val[1]
            sys.exit()
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


with open('schillingData.txt', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

for xInput, yInput in your_list:
    xTest.append(mapProteinToInt(xInput))
    yTest.append([int(mapProteinSequenceResult(yInput)), (-1*int(mapProteinSequenceResult(yInput)))])

count =0
goodResults=0
def roundResult(input):
    if input > 0.5:
        return 1
    else:
        return 0

sumPosValues = 0
sumNegValues = 0
posCount = 0
negCount = 0

for x_test in xTrain:
    nnResult = sess.run(y, feed_dict={x: [x_test]})
    roundedNNResult = roundResult(nnResult[0][0]) # will be a value either 1 or 0

    # actualResult = yTest[count][0] # will be a value either 1 or zero
    actualResult = yTrain[count][0] # will be a value either 1 or zero

    if roundedNNResult == actualResult:
        goodResults +=1

    if actualResult >= 1:
        # print " Protein sequence: ",x_test," Actual: ",actualResult, " Result: ", nnResult[0][0], " Rounded ",roundedNNResult
        sumPosValues+= nnResult[0][0]
        posCount+=1
    else:
        sumNegValues+= nnResult[0][0]
        negCount+=1


    count+=1

print "Correct predictions: ", goodResults
print "Total tests: ", count
print "Percentage Correct", float(goodResults)/count
print "Average Pos value: ", float(sumPosValues)/posCount
print "Average Neg value: ", float(sumNegValues)/negCount
print "Number of pos", posCount
print "Number of neg", negCount



