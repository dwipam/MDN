import tensorflow as tf
import tensorflow_probability as tfp

class Network:
	def __init__(self,x,y, K=50):
		self.layer_1 = tf.layers.dense(x, units=50, activation=tf.nn.tanh, name="layer_1")
		self.layer_2 = tf.layers.dense(self.layer_1, units=20, activation=tf.nn.tanh)
		self.layer_2 = tf.layers.dense(self.layer_2, units=20, activation=tf.nn.tanh)
		self.layer_2 = tf.layers.dense(self.layer_2, units=20, activation=tf.nn.tanh)
		self.layer_2 = tf.layers.dense(self.layer_2, units=20, activation=tf.nn.tanh, name="layer_2")
		self.mu = tf.layers.dense(self.layer_2, units=K, activation=None, name="mu")
		self.var = tf.exp(tf.layers.dense(self.layer_2, units=K, activation=None, name="sigma"))
		self.pi = tf.layers.dense(self.layer_2, units=K, activation=tf.nn.softmax, name="mixing")

		self.likelihood = tfp.distributions.Normal(loc=self.mu, scale=self.var)
		self.out = self.likelihood.prob(y)
		self.out = tf.multiply(self.out, self.pi)
		self.out = tf.reduce_sum(self.out, 1, keepdims=True)
		self.out = -tf.log(self.out + 1e-10)
		self.mean_loss = tf.reduce_mean(self.out)

		self.global_step = tf.Variable(0, trainable=False)
		self.learning_rate = tf.train.exponential_decay(0.015, self.global_step, 100, .99, staircase=True,)
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss,global_step=self.global_step)
		self.init = tf.global_variables_initializer()
		self.init_l = tf.local_variables_initializer()

		# Initialize coefficients
		self.sess = tf.Session()
		self.sess.run(self.init)
		self.sess.run(self.init_l)
