import tensorflow as tf
import numpy as np
from scipy import special

from nb_vae import NegativeBinomialVAE


class NegativeBinomialVAEb(NegativeBinomialVAE):

    def _log_likelihood(self, h_r, h_p):

        temp = tf.exp(-tf.multiply(tf.exp(h_r), tf.log(tf.exp(h_p) + 1)))

        temp = tf.clip_by_value(temp, 1e-5, 1 - 1e-5)
        ll = tf.multiply(self.input_ph, tf.log(1 - temp))

        ll += tf.multiply(1 - self.input_ph, tf.log(temp))

        return ll

    def get_predictive_rate(self, h_r, h_p, test_data):

        l_prime = 1 - np.power(special.expit(-h_p), np.exp(h_r))

        return l_prime





