import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

'''
    手写数字识别
'''
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def add_layer(input,in_size,out_size,activate_func=None):
    Weight=tf.Variable(tf.random_uniform([in_size,out_size]))
    baise=tf.Variable(tf.zeros([1,out_size])+0.1)
    prdict_y=tf.matmul(input,Weight)+baise
    if activate_func==None:
        output=prdict_y
    else:
        output=activate_func(prdict_y)
    return output

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={X: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={X: v_xs, Y: v_ys})
    return result

#输入层
X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])

#输出层
predict_layer=add_layer(X,784,10,activate_func=tf.nn.softmax)

#损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(predict_layer),reduction_indices=[1]))

#优化
train=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for step in range(1000001):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train,feed_dict={X:batch_xs,Y:batch_ys})
    if step%50==0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))