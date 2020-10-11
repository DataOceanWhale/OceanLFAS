#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import zhusuan as zs
from zhusuan.utils import merge_dicts
from ..third_party import zs_sgmcmc


class BNNRegressor:
  def __init__(self, x_dim, hidden_layer_sizes=[50,], n_particles=20, learning_rate=4e-5, optimizer='sgld', epochs=1000, verbose=False, update_logstd=True):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.n_particles = n_particles
    self.learning_rate = learning_rate
    self.optimizer = optimizer
    self.epochs = epochs
    self.verbose = verbose
    self.update_logstd = update_logstd
    self.sess = tf.Session()
    self._define_model(x_dim)

  def _define_model(self, x_dim):
    y_logstd = -1.

    @zs.meta_bayesian_net(scope="bnn", reuse_variables=True)
    def build_bnn(x, layer_sizes, logstds, n_particles):
        bn = zs.BayesianNet()
        h = tf.tile(x[None, ...], [n_particles, 1, 1])
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            w = bn.normal("w" + str(i), tf.zeros([n_out, n_in + 1]),
                          logstd=logstds[i], group_ndims=2, n_samples=n_particles)
            h = tf.concat([h, tf.ones(tf.shape(h)[:-1])[..., None]], -1)
            h = tf.einsum("imk,ijk->ijm", w, h) / tf.sqrt(
                tf.cast(tf.shape(h)[2], tf.float32))
            if i < len(layer_sizes) - 2:
                h = tf.nn.relu(h)

        y_mean = bn.deterministic("y_mean", tf.squeeze(h, 2))
        bn.normal("y", y_mean, logstd=y_logstd)
        return bn

    update_logstd = self.update_logstd

    x = tf.placeholder(tf.float32, shape=[None, x_dim])
    self.x = x
    y = tf.placeholder(tf.float32, shape=[None])
    time_decay = tf.placeholder(tf.float32, shape=[None])
    layer_sizes = [x_dim] + self.hidden_layer_sizes + [1]
    w_names = ["w" + str(i) for i in range(len(layer_sizes) - 1)]
    wv = []
    logstds = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        wv.append(tf.Variable(
            tf.random_uniform([self.n_particles, n_out, n_in + 1])*4-2))

    if update_logstd:
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            logstds.append(tf.Variable(tf.zeros([n_out, n_in + 1])))
    else:
        logstds = [20., 20., 20., 20.]
        print("No prior!")

    model = build_bnn(x, layer_sizes, logstds, self.n_particles)

    def log_joint(bn):
        log_pws = bn.cond_log_prob(w_names)
        log_py_xw = bn.cond_log_prob('y')
        return tf.add_n(log_pws) + tf.reduce_sum(time_decay * log_py_xw, 1)

    model.log_joint = log_joint

    if self.optimizer == 'sgld':
        sgmcmc = zs_sgmcmc.SGLD(learning_rate=self.learning_rate, add_noise=True)
    elif self.optimizer == 'sghmc':
        sgmcmc = zs_sgmcmc.SGHMC(learning_rate=self.learning_rate, friction=0.2, n_iter_resample_v=1000,
                          second_order=True)
    latent = dict(zip(w_names, wv))
    observed = {'y': y}

    # E step: Sample the parameters
    sample_op, sgmcmc_info = sgmcmc.sample(model, observed=observed,
                                           latent=latent)

    if update_logstd:
      # M step: Update the logstd hyperparameters
      esti_logstds = [0.5*tf.log(tf.reduce_mean(w*w, axis=0)) for w in wv]
      output_logstds = dict(zip(w_names,
                                [0.5*tf.log(tf.reduce_mean(w*w)) for w in wv]))
      assign_ops = [logstds[i].assign(logstd)
                    for (i, logstd) in enumerate(esti_logstds)]
      assign_op = tf.group(assign_ops)

    # prediction: rmse & log likelihood
    bn = model.observe(**merge_dicts(latent, observed))
    y_mean = bn["y_mean"]
    self.y_pred = tf.reduce_mean(y_mean, 0)
    # TODO: check it
    self.a_std = np.exp(y_logstd)
    self.e_std = tf.sqrt(tf.reduce_mean((y_mean - self.y_pred) ** 2, 0))
    self.nll = tf.reduce_mean((self.y_pred - y) ** 2)

    self.sample_op = sample_op
    self.x = x
    self.y = y
    self.time_decay = time_decay
    if update_logstd:
      self.assign_op = assign_op
    self.update_logstd = update_logstd
    
  
  def fit(self, x_train, y_train, time_decay=None):
    # Run the inference
    self.sess.run(tf.global_variables_initializer())
    if time_decay is None:
      time_decay = np.ones([x_train.shape[0]])
    for epoch in range(1, self.epochs + 1):
        self.sess.run(self.sample_op, feed_dict={self.x: x_train, self.y: y_train, self.time_decay: time_decay})
        if self.verbose:
          if epoch % 50 == 0:
            print(self.sess.run(self.nll, feed_dict={self.x: x_train, self.y: y_train, self.time_decay: time_decay}))
        if self.update_logstd:
            self.sess.run(self.assign_op)

  def predict(self, x_test, return_std=False):
      y_mean, e_std = self.sess.run([self.y_pred, self.e_std], feed_dict={self.x: x_test})
      a_std = self.a_std * np.ones_like(e_std)
      if not return_std:
          return y_mean
      else:
          return y_mean, np.transpose((a_std, e_std))
