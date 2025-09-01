# -*- coding: utf-8 -*-

import os
import json
import tensorflow as tf

a = tf.constant([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])   #3*2
b = tf.constant([[1, 0], [2, 1], [0, 1]])  #3*2
c = tf.nn.embedding_lookup(a, b)
d = tf.reduce_mean(c, axis=1)
e = tf.concat([d, a], 1)

with tf.Session() as sess:
    print(c)
    print(sess.run(c))
    print(d)
    print(sess.run(d))
    print(e)
    print(sess.run(e))

