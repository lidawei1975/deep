#!/usr/bin/env python


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math as m


tf.compat.v1.enable_eager_execution()



class MyLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(MyLayer, self).__init__()
        self.n_unit_softmax = 3
      
        

    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(input_shape[-1], self.n_unit_softmax),
                                initializer='random_normal',
                                trainable=True,name='lastlayer_w1')

        self.b1 = self.add_weight(shape=(1,self.n_unit_softmax),
                                initializer='random_normal',
                                trainable=True,name='lastlayer_b1')


    def call(self, inputs):
        df=tf.matmul(inputs, self.w1) + self.b1
        # df=tf.nn.softmax(df)
        return df





with open('coor.xvg', 'r') as file:
    x=np.loadtxt(file)
x_train = x.astype('float32')

with open('flag.xvg', 'r') as file:
    y=np.loadtxt(file)
y_train = y.astype('float32')


with open('./validation_set/coor.xvg', 'r') as file:
    x=np.loadtxt(file)
x_test = x.astype('float32')

with open('./validation_set/flag.xvg', 'r') as file:
    y=np.loadtxt(file)
y_test = y.astype('float32')


s=x_train.shape
ss=x_test.shape
x_train=tf.reshape(x_train, [s[0], s[1], 1])
y_train=tf.reshape(y_train, [int(round(y_train.shape[0]/s[1])), s[1],9])
x_test=tf.reshape(x_test, [ss[0], ss[1], 1])
y_test=tf.reshape(y_test, [int(round(y_test.shape[0]/ss[1])), ss[1],9])


log_file = open('train_log.txt', 'a') 
print(x_train.shape)
print(x_train.shape,file=log_file)
print(y_train.shape)
print(y_train.shape,file=log_file)
print(x_test.shape)
print(x_test.shape,file=log_file)
print(y_test.shape)
print(y_test.shape,file=log_file)


num_hot=3

myinput = tf.keras.layers.Input(shape=(200,1))
conv1=tf.keras.layers.Conv1D(filters=40,kernel_size=11,padding='same',strides=1,activation=tf.nn.relu,input_shape=(200,1))(myinput)
conv2=tf.keras.layers.Conv1D(filters=20,kernel_size=1,padding='same',strides=1,activation=tf.nn.relu)(conv1)
convs2=tf.keras.layers.Conv1D(filters=10,kernel_size=11,padding='same',strides=1,activation=tf.nn.relu)(conv2)
convs3=tf.keras.layers.Conv1D(filters=20,kernel_size=1,padding='same',strides=1,activation=tf.nn.relu)(convs2)
conv3=tf.keras.layers.Conv1D(filters=10,kernel_size=1,padding='same',strides=1,activation=tf.nn.relu)(convs3)
conv4=tf.keras.layers.Conv1D(filters=30,kernel_size=11,padding='same',strides=1,activation=tf.nn.relu)(conv3)
conv5=tf.keras.layers.Conv1D(filters=18,kernel_size=1,padding='same',strides=1,activation=tf.nn.relu)(conv4)
conv6=tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(conv5)
softmax1=MyLayer()(conv6)
conv8=tf.keras.layers.Conv1D(filters=8,kernel_size=1,padding='same',strides=1,activation=tf.keras.activations.linear)(conv5)
myoutput=tf.keras.layers.concatenate([softmax1, conv8])
model2 = tf.keras.Model(inputs=myinput, outputs=myoutput)

model2.build(input_shape=(1024,160,1))
print(model2.summary())

for layer in model2.layers:
    print(layer.output_shape)


model=model2

#model.load_weights('./peak_weight')
#print("load weight from peak_weight")
#print("load weight from peak_weight",file=log_file)

batch_size = 10000
# batch_size = 1
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)


lr=0.002
print('learning rate is ',lr)
print('learning rate is ',lr,file=log_file)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


for epoch in range(4000):

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            result = model(x_batch_train)    
            result2=tf.slice(result,[0,0,0],[result.shape[0],result.shape[1],num_hot])
            y_batch_train1=tf.slice(y_batch_train,[0,0,0],[y_batch_train.shape[0],y_batch_train.shape[1],1])
            y_batch_train2=tf.reshape(y_batch_train1,[y_batch_train.shape[0],y_batch_train.shape[1]])
            y_batch_train2=tf.dtypes.cast(y_batch_train2,tf.int32)
            y_batch_train3=tf.one_hot(y_batch_train2, depth=num_hot,axis=-1)

            # print('target1=',y_batch_train1)
            # print('target2=',y_batch_train2)
            # print('target3=',y_batch_train3)

            sample_weights = tf.reduce_sum(tf.multiply(y_batch_train3, [0.4,1.4,0.9]), 2)
            loss_value0=tf.compat.v1.losses.softmax_cross_entropy(y_batch_train3,result2,weights=sample_weights)

            target_reg1=tf.slice(y_batch_train,[0,0,1],[y_batch_train.shape[0],y_batch_train.shape[1],4])
            result1=tf.slice(result,[0,0,num_hot],[result.shape[0],result.shape[1],4])
            mask1=tf.dtypes.cast(tf.math.logical_and(tf.greater(y_batch_train1,0.5),tf.less(y_batch_train1,1.5)),tf.float32)
            loss_value1 = tf.reduce_mean(tf.square(tf.math.multiply(result1-target_reg1,mask1)))

            target_reg2=tf.slice(y_batch_train,[0,0,5],[y_batch_train.shape[0],y_batch_train.shape[1],4])
            result2=tf.slice(result,[0,0,num_hot+4],[result.shape[0],result.shape[1],4])
            mask2=tf.dtypes.cast(tf.greater(y_batch_train1,1.5),tf.float32)
            loss_value2 = tf.reduce_mean(tf.square(tf.math.multiply(result2-target_reg2,mask2)))


            #print('target_reg=',target_reg)
            #print('result1=',result1)
            #print('mask=',mask)
            #print('loss2=',loss_value2)


            loss_value = loss_value0 + loss_value1*5 + loss_value2*15
            # print('step=',step)
            # print('loss=',loss_value)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
       
        # if step % 10 == 0:
        #     print("Epoch {:03d}: Batch: {:03d} Loss: {:.3f}".format(epoch,step,loss_value))


    for step, (x_batch_train, y_batch_train) in enumerate(test_dataset):

       
        result = model(x_batch_train)    
        result2=tf.slice(result,[0,0,0],[result.shape[0],result.shape[1],num_hot])
        y_batch_train1=tf.slice(y_batch_train,[0,0,0],[y_batch_train.shape[0],y_batch_train.shape[1],1])
        y_batch_train2=tf.reshape(y_batch_train1,[y_batch_train.shape[0],y_batch_train.shape[1]])
        y_batch_train2=tf.dtypes.cast(y_batch_train2,tf.int32)
        y_batch_train3=tf.one_hot(y_batch_train2, depth=num_hot,axis=-1)

          

        sample_weights = tf.reduce_sum(tf.multiply(y_batch_train3, [0.4,1.4,0.9]), 2)
        tloss_value0=tf.compat.v1.losses.softmax_cross_entropy(y_batch_train3,result2,weights=sample_weights)

        target_reg1=tf.slice(y_batch_train,[0,0,1],[y_batch_train.shape[0],y_batch_train.shape[1],4])
        result1=tf.slice(result,[0,0,num_hot],[result.shape[0],result.shape[1],4])
        mask1=tf.dtypes.cast(tf.math.logical_and(tf.greater(y_batch_train1,0.5),tf.less(y_batch_train1,1.5)),tf.float32)
        tloss_value1 = tf.reduce_mean(tf.square(tf.math.multiply(result1-target_reg1,mask1)))

        target_reg2=tf.slice(y_batch_train,[0,0,5],[y_batch_train.shape[0],y_batch_train.shape[1],4])
        result2=tf.slice(result,[0,0,num_hot+4],[result.shape[0],result.shape[1],4])
        mask2=tf.dtypes.cast(tf.greater(y_batch_train1,1.5),tf.float32)
        tloss_value2 = tf.reduce_mean(tf.square(tf.math.multiply(result2-target_reg2,mask2)))


        tloss_value = tloss_value0 + tloss_value1*5 + tloss_value2*15

    if epoch % 50 ==0 or epoch==10 or epoch==20 :
        print("Epoch {:03d}, after all batch, Loss is {:.8f} {:.8f} {:.8f} {:.8f} and {:.8f} {:.8f} {:.8f} {:.8f} ".format(epoch,loss_value,loss_value0,loss_value1,loss_value2,tloss_value,tloss_value0,tloss_value1,tloss_value2))
        print("Epoch {:03d}, after all batch, Loss is {:.8f} {:.8f} {:.8f} {:.8f} and {:.8f} {:.8f} {:.8f} {:.8f} ".format(epoch,loss_value,loss_value0,loss_value1,loss_value2,tloss_value,tloss_value0,tloss_value1,tloss_value2), file=log_file)
        log_file.flush()
        model.save_weights('./peak_weight')

model.save_weights('./peak_weight')

