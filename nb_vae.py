import numpy as np
import tensorflow as tf
from scipy import special


class NegativeBinomialVAE():

    def __init__(self, arch, lr=1e-3, random_seed=None):

        self.decoder_arch = arch

        self.encoder_arch = arch[::-1]

        self.lr = lr
        self.random_seed = random_seed

        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, arch[-1]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

    def _log_likelihood(self, h_r, h_p):

        ll = tf.lgamma(tf.exp(h_r) + self.input_ph) - tf.lgamma(tf.exp(h_r))
        ll += h_p * self.input_ph - tf.log(tf.exp(h_p) + 1) * (self.input_ph + tf.exp(h_r))

        return ll

    def _encoder_pass(self):

        mu_z, std_z, kl = None, None, None
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        for i, (w, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            h = tf.matmul(h, w) + b

            if i != len(self.encoder_weights) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_z = h[:, :self.encoder_arch[-1]]
                logvar_q = h[:, self.encoder_arch[-1]:]
                std_z = tf.exp(0.5 * logvar_q)
                kl = tf.reduce_sum(
                    0.5 * (-logvar_q + tf.exp(logvar_q) + mu_z ** 2 - 1), axis=1)
        return mu_z, std_z, kl

    def _decoder_pass_r(self, z):

        h_r = z
        for i, (w, b) in enumerate(zip(self.decoder_weights_r, self.decoder_biases_r)):
            h_r = tf.matmul(h_r, w) + b
            if i != len(self.decoder_weights_r) - 1:
                h_r = tf.nn.tanh(h_r)

        return h_r

    def _decoder_pass_p(self, z):

        h_p = z
        for i, (w, b) in enumerate(zip(self.decoder_weights_p, self.decoder_biases_p)):
            h_p = tf.matmul(h_p, w) + b
            if i != len(self.decoder_weights_p) - 1:
                h_p = tf.nn.tanh(h_p)

        return h_p

    def build_graph(self):

        self._construct_encoder_weights()
        self._construct_decoder_weights_r()
        self._construct_decoder_weights_p()

        saver = tf.train.Saver()

        mu_z, std_z, kl = self._encoder_pass()
        epsilon = tf.random_normal(tf.shape(std_z))
        z = mu_z + self.is_training_ph * epsilon * std_z

        h_r = self._decoder_pass_r(z)
        h_p = self._decoder_pass_p(z)

        ll = self._log_likelihood(h_r, h_p)
        neg_ll = -tf.reduce_mean(tf.reduce_sum(ll, axis=-1))
        neg_elbo = neg_ll + self.anneal_ph * tf.reduce_mean(kl)

        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_elbo)

        return saver, train_op, h_r, h_p

    def _construct_encoder_weights(self):

        self.encoder_weights, self.encoder_biases = [], []

        for i, (d_in, d_out) in enumerate(zip(self.encoder_arch[:-1], self.encoder_arch[1:])):
            if i == len(self.encoder_arch[:-1]) - 1:
                d_out *= 2

            weight_key = "encoder_weights_%d" % i
            bias_key = "encoder_bias_%d" % i
            self.encoder_weights.append(tf.get_variable(name=weight_key, shape=[d_in, d_out],
                                                        initializer=tf.contrib.layers.xavier_initializer(
                                                            seed=self.random_seed)))
            self.encoder_biases.append(tf.get_variable(name=bias_key, shape=[d_out],
                                                       initializer=tf.truncated_normal_initializer(
                                                           stddev=0.001, seed=self.random_seed)))

        self.decoder_weights_r, self.decoder_biases_r = [], []

    def _construct_decoder_weights_r(self):

        for i, (d_in, d_out) in enumerate(zip(self.decoder_arch[:-1], self.decoder_arch[1:])):
            weight_key = "decoder_weights_r_%d" % i
            bias_key = "decoder_bias_r_%d" % i
            self.decoder_weights_r.append(tf.get_variable(name=weight_key, shape=[d_in, d_out],
                                                          initializer=tf.contrib.layers.xavier_initializer(
                                                              seed=self.random_seed)))
            self.decoder_biases_r.append(tf.get_variable(name=bias_key, shape=[d_out],
                                                         initializer=tf.truncated_normal_initializer(
                                                             stddev=0.001, seed=self.random_seed)))

    def _construct_decoder_weights_p(self):

        self.decoder_weights_p, self.decoder_biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.decoder_arch[:-1], self.decoder_arch[1:])):
            weight_key = "decoder_weights_p_%d" % i
            bias_key = "decoder_bias_p_%d" % i
            self.decoder_weights_p.append(tf.get_variable(name=weight_key, shape=[d_in, d_out],
                                                          initializer=tf.contrib.layers.xavier_initializer(
                                                              seed=self.random_seed)))
            self.decoder_biases_p.append(tf.get_variable(name=bias_key, shape=[d_out],
                                                         initializer=tf.truncated_normal_initializer(
                                                             stddev=0.001, seed=self.random_seed)))

    def get_predictive_rate(self, h_r, h_p, test_data):

        l_prime = np.multiply(test_data + np.exp(h_r), special.expit(h_p))

        return l_prime

