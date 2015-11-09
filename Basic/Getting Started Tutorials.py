import tensorflow as tf


print "-------Session----------"
#sets up the problem
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[3.], [3.]])
product = tf.matmul(matrix1,matrix2)



#computes the result to the problem
session = tf.Session()
with tf.device("/cpu:2"):
	result = session.run(product)
print result
session.close()


print "-------Interactive Session------"

intSession = tf.InteractiveSession()

x = tf.Variable([4.0,3.0])
a = tf.constant([2.0,1.0])

x.initializer.run()

subtact = tf.sub(x,a)
print subtact.eval()



print "-------Variables----------"
#making a variable using tensorflow
var = tf.Variable(0,name="counter")
one = tf.constant(1)
new_value = tf.add(var,one) 
update = tf.assign(var,new_value)

#initializing the variables before using them

init_op = tf.initialize_all_variables()

with tf.Session() as sees:
	sees.run(init_op)
	print sees.run(var)

	for _ in range(3):
		sees.run(update)
		print sees.run(var)

print "-------Fetch-----------"
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print result

print "-----Feeds---------"
input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})



