import tensorflow as tf
import numpy as np

'''
   线性回归求解
'''
X_data=np.random.rand(100).astype(np.float32)
Y_data=0.3*X_data+0.1

W=tf.Variable(tf.random_uniform([1],-1,1))
B=tf.Variable(tf.zeros([1]))

Y=W*X_data+B
loss=tf.reduce_mean(tf.square(Y-Y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

session=tf.Session()
session.run(init)
for step in range(201):
    session.run(train)
    if step%20==0:
        print(step,session.run(W),session
              .run(B))