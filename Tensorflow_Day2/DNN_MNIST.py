# Lab 10 MNIST and Dropout
import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)#读取输入数据
learning_rate=0.001
training_epochs=15
batch_size=100

#定义输入数据placeholder
X=tf.placeholder(tf.float32,[None,28*28])
y=tf.placeholder(tf.float32,[None,10])
#dropout中的概率placeholder
keep_prob=tf.placeholder(tf.float32)

#定义网络的各层
#1.第一层，将特征从784降到512

W1=tf.get_variable(name='W1',shape=[784,512],initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.random_normal([512]))
L1=tf.nn.relu(tf.matmul(X,W1)+b1)#第一层输出
L1=tf.nn.dropout(L1,keep_prob)#加了dropout之后的输出

#第二层，维持512个特征不变
W2=tf.get_variable('W2',shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
b2=tf.Variable(tf.random_normal([512]))
L2=tf.nn.relu(tf.matmul(L1,W2)+b2)#第一层输出
L2=tf.nn.dropout(L2,keep_prob)#加了dropout之后的输出
#第三层，维持512个特征不变

W3=tf.get_variable('W3',shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
b3=tf.Variable(tf.random_normal([512]))
L3=tf.nn.relu(tf.matmul(L2,W3)+b3)#第一层输出
L3=tf.nn.dropout(L3,keep_prob)#加了dropout之后的输出

#第四层，维持512个特征不变，可以算到softmax层去？
W4=tf.get_variable('W4',shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
b4=tf.Variable(tf.random_normal([512]))
L4=tf.nn.relu(tf.matmul(L3,W4)+b4)#第一层输出
L4=tf.nn.dropout(L4,keep_prob)#加了dropout之后的输出
#第五层，降低到10个特征，即softmax输出层
#第三层，维持512个特征不变

W5=tf.get_variable('W5',shape=[512,10],initializer=tf.contrib.layers.xavier_initializer())
b5=tf.Variable(tf.random_normal([10]))
hypothesis=tf.matmul(L4,W5)+b5#第一层输出
#计算cost
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#创建会话
sess=tf.Session()
sess.run(tf.global_variables_initializer())#运行这个方法设置全局变量

for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        feed_dict={X:batch_xs,y:batch_ys,keep_prob:0.7}
        cost_val,_=sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost+=cost_val/total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))#每一轮结束都打印cost

print("Training Finished!")

correct_prediction=tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
accuracy=tf.cast(correct_prediction,tf.float32)
accuracy=tf.reduce_mean(accuracy)

print("Accuracy: {}".format(sess.run(accuracy,feed_dict={X:mnist.test.images,y:mnist.test.labels,keep_prob:1})))

r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))