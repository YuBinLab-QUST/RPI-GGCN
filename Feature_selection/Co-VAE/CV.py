# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:08:40 2023

@author: dell
"""


# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# import tensorflow._api.v2.compat.v1 as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters*2,kernel_size=k_size, stride=1, padding=k_size//2),
            
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size//2),
            
        )
       
        # self.conv4 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size//2),
            
        # )

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        output = self.out(x)
        output = output.squeeze()
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2

class Autoencoder(object):

    def __init__(self, n_input, n_hidden1,n_hidden2,n_hidden3, transfer_function=tf.nn.softplus, optimizer = tf.compat.v1.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden1 = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.hidden2 = self.transfer(tf.add(tf.matmul(self.hidden1, self.weights['w2']), self.weights['b2']))
        self.hidden3 = self.transfer(tf.add(tf.matmul(self.hidden2, self.weights['w3']), self.weights['b3']))
        self.reconstruction = tf.add(tf.matmul(self.hidden3, self.weights['w4']), self.weights['b4'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden1],
            initializer=tf.glorot_uniform_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden1], dtype=tf.float32))
        all_weights['w2'] = tf.get_variable("w2", shape=[self.n_hidden1, self.n_hidden2],
            initializer=tf.glorot_uniform_initializer())
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_hidden2], dtype=tf.float32))
        all_weights['w3'] = tf.get_variable("w3", shape=[self.n_hidden2, self.n_hidden3],
            initializer=tf.glorot_uniform_initializer())
        all_weights['b3'] = tf.Variable(tf.zeros([self.n_hidden3], dtype=tf.float32))
        all_weights['w4'] = tf.Variable(tf.zeros([self.n_hidden3, self.n_input], dtype=tf.float32))
        all_weights['b4'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden2, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'],self.weights['w2'],self.weights['w3'],self.weights['w4'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'],self.weights['b2'],self.weights['b3'],self.weights['b4'])

