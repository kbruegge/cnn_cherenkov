import tensorflow as tf
import numpy as np


y_true = np.random.uniform(0, 1, size=(10, 6))
y_pred = np.random.uniform(0, 1, size=(10, 2))

print(y_true)
print(np.count_nonzero(y_true < 0.2))

on_region_radius_degree = 0.2
alpha = 0.2

theta = tf.convert_to_tensor(y_true, dtype=tf.float32)
y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

N_on = tf.count_nonzero(theta[:, 0] < on_region_radius_degree, dtype=tf.float32)
N_off = tf.reduce_sum([tf.count_nonzero(theta[:, i] < on_region_radius_degree, dtype=tf.float32) for i in range(1, 6)])

S = (N_on - alpha * N_off) / tf.sqrt(N_on + alpha**2 * N_off,)
loss = 100.0 - S

sess = tf.Session()
print(sess.run(loss))

#
# def region_loss(y_pred, y_true):
#     with tf.name_scope(None):
#         # distance_on = y_true[:, 0]
#         # a = tf.multiply(y_pred, distance_on)
#         # b_1 = tf.multiply(tf.subtract(1.0, y_pred), y_true[:, 1])
#         return tf.multiply(y_pred, y_true)
