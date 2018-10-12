import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


'''
   神经网络求解二次函数
'''

# 定义神经网络
def add_layer(input,in_size,out_size,activate_func=None):
    Weight=tf.Variable(tf.random_uniform([in_size,out_size]))
    baisc=tf.Variable(tf.zeros([1,out_size])+0.1)
    y=tf.matmul(input,Weight)+baisc
    if activate_func==None:
        output=y
    else:
        output=activate_func(y)
    return output

# 准备数据
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 输入层
X=tf.placeholder(tf.float32,[None,1])
Y=tf.placeholder(tf.float32,[None,1])

# hidden 层
l1=add_layer(X,1,10,activate_func=tf.nn.relu)

# output 层
output_layer=add_layer(l1,10,1,activate_func=None)

#定义loss
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y_data - output_layer),
                     reduction_indices=[1]))

#定义优化方式
optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for step in range(10000):
    sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
    if step%50==0:
        try:
            ax.lines.remove(lines[0])
        except Exception as e:
            pass
        prdiction_value=sess.run(output_layer,feed_dict={X:x_data})
        print(prdiction_value)
        lines=ax.plot(x_data,prdiction_value,color='red',lw=5)
        plt.pause(1)
        # print(step,sess.run(loss,feed_dict={X:x_data,Y:y_data}))