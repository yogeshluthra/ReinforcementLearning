import os
from collections import deque

import matplotlib.pyplot as plt
import tensorflow as tf


class MultiLayerNN(object):
    def __init__(self, input_dim, output_dim, learning_rate=0.001, beta=0.01, summaries_dir='./summary', scope='test'):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            # self.session = tf.InteractiveSession()
            # self.session.run(tf.global_variables_initializer())
            self.learning_rate = learning_rate;
            self.beta = beta
            self.memory = deque()
            self.input_dim = input_dim
            self.States = tf.placeholder(tf.float32, shape=[None, self.input_dim])
            self.output_dim = output_dim
            self.selOutputs = tf.placeholder(tf.float32, shape=[None, self.output_dim])
            self.targetVals = tf.placeholder(tf.float32, shape=[None])
            (self.estimations,
             self.optimizer,
             self.loss,
             self.losses,
             self.W_fcfinal,
             self.b_fcfinal) = self.build_network()

            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)
            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("loss_hist", self.losses),
                tf.summary.histogram("q_values_hist", self.estimations),
                tf.summary.scalar("max_q_value", tf.reduce_max(self.estimations)),
                tf.summary.histogram("final_layer_weights", self.W_fcfinal),
                tf.summary.histogram("final_layer_biases", self.b_fcfinal)
            ])
            self.lossTracker = []

    def formWeights(self, shape, stddev=0.5):
        W = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(W)

    def formBiases(self, shape, value=0.5):
        B = tf.constant(value, shape=shape)
        return tf.Variable(B)

    def get_Weights_Biases(self, shape):
        return self.formWeights(shape), self.formBiases(shape[-1:])

    def build_network(self):
        """
        3 fully connected layers
        First 2 layers activation=ReLU
        Last Layer is simply linear
        """
        l1_outputs = 50  # earlier 1000 really bad.. earlier 500.. earlier 200. seemed ok but doubt on capacity to learn.
        l2_outputs = 50  # earlier 1000 really bad.. earlier 500.. earlier 200. seemed ok but doubt on capacity to learn.
        l3_outputs = 50
        # output at final layer, with dimension = self.output_dim

        # 1st fully connected layer with ReLU activation. 100 hidden units
        W_fc1, b_fc1 = self.get_Weights_Biases([self.input_dim, l1_outputs])
        h_fc1 = tf.nn.relu(tf.matmul(self.States, W_fc1) + b_fc1)
        regularizer = tf.nn.l2_loss(W_fc1)

        #         h_fc1_flat=tf.reshape(h_fc1, [-1, l1_outputs])
        # 2nd fully connected layer with ReLU activation. 100 hidden units
        W_fc2, b_fc2 = self.get_Weights_Biases([l1_outputs, l2_outputs])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        regularizer = regularizer + tf.nn.l2_loss(W_fc2)

        # # 3rd fully connected layer with ReLU activation. 100 hidden units
        # W_fc3, b_fc3 = self.get_Weights_Biases([l2_outputs, l3_outputs])
        # h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        # regularizer = regularizer + tf.nn.l2_loss(W_fc3)

        h_ToFinal = h_fc2
        hfinal_dim = l2_outputs
        #         h_fc2_flat=tf.reshape(h_fc2, [-1, l2_outputs])
        # final fully connected LINEAR layer.
        W_fcfinal, b_fcfinal = self.get_Weights_Biases([hfinal_dim, self.output_dim])
        estimations = tf.matmul(h_ToFinal, W_fcfinal) + b_fcfinal
        regularizer = regularizer + tf.nn.l2_loss(W_fcfinal)

        estimated_actionVals = tf.reduce_sum(tf.mul(estimations, self.selOutputs), axis=1)
        losses = tf.squared_difference(self.targetVals, estimated_actionVals)
        loss = tf.reduce_mean(losses + self.beta * regularizer)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, global_step=self.global_step)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6).minimize(loss, global_step=self.global_step)
        #         optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss, global_step=self.global_step)

        return estimations, optimizer, loss, losses, W_fcfinal, b_fcfinal

    def train(self, sess, targets, selOutputs, states):
        #         _, summaries, loss, global_step = sess.run([
        #             self.optimizer, self.summaries, self.loss, self.global_step],
        #                                                    feed_dict={self.targetVals: targets,
        #                                                       self.selOutputs: selOutputs,
        #                                                       self.States: states})
        #         if self.summary_writer:
        #             self.summary_writer.add_summary(summaries, global_step)

        _, loss, global_step = sess.run([
            self.optimizer, self.loss, self.global_step],
            feed_dict={self.targetVals: targets,
                       self.selOutputs: selOutputs,
                       self.States: states})

        #         _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.targetVals: targets,
        #                                                                   self.selOutputs: selOutputs,
        #                                                                   self.States: states})
        self.lossTracker.append(loss)

    def getEstimatesFor(self, sess, states):
        loc_states = [states] if len(states.shape) == 1 else states
        return sess.run(self.estimations, feed_dict={self.States: loc_states})

    def plotLoss(self):
        plt.plot(self.lossTracker)
        plt.title('loss')
        plt.show()
