import tensorflow as tf
import tensorflow_probability as tfp

class Network:
	def __init__(self, x, y, K=50):
		self.x = x
		self.layer_1 = tf.layers.dense(x, units=50, activation=tf.nn.tanh, name="layer_1")
		self.layer_2 = tf.layers.dense(self.layer_1, units=20, activation=tf.nn.tanh)
		self.layer_2 = tf.layers.dense(self.layer_2, units=20, activation=tf.nn.tanh)
		self.layer_2 = tf.layers.dense(self.layer_2, units=20, activation=tf.nn.tanh)
		self.layer_2 = tf.layers.dense(self.layer_2, units=20, activation=tf.nn.tanh, name="layer_2")
		self.mu = tf.layers.dense(self.layer_2, units=K, activation=None, name="mu")
		self.var = tf.exp(tf.layers.dense(self.layer_2, units=K, activation=tf.nn.softplus, name="sigma"))
		self.pi = tf.layers.dense(self.layer_2, units=K, activation=tf.nn.softmax, name="mixing")

		# -------------------- Not using Mixture Family ------------------------
		# self.likelihood = tfp.distributions.Normal(loc=self.mu, scale=self.var)
		# self.out = self.likelihood.prob(y)
		# self.out = tf.multiply(self.out, self.pi)
		# self.out = tf.reduce_sum(self.out, 1, keepdims=True)
		# self.out = -tf.log(self.out + 1e-10)
		# self.mean_loss = tf.reduce_mean(self.out)

		# -------------------- Not using Mixture Family ------------------------
		self.mixture_distribution = tfp.distributions.Categorical(probs=self.pi)
		self.distribution = fp.distributions.Normal(loc=self.mu, scale=self.var)
		self.likelihood = tfp.distributions.MixtureSameFamily(
															mixture_distribution=self.mixture_distribution,
        													components_distribution=self.distribution)
		self.log_likelihood = -self.likelihood.log_prob(tf.transpose(y))
		self.mean_loss = tf.reduce_mean(self.log_likelihood)
		# ----------------------------------------------------------------------

		self.global_step = tf.Variable(0, trainable=False)
		self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.8).minimize(self.mean_loss)
		self.init = tf.global_variables_initializer()

		# Initialize coefficients
		self.sess = tf.Session()
		self.sess.run(self.init)

class Network1Comp:
	def __init__(self, x, y):
		self.x = x
		self.layer_1 = tf.layers.dense(x, units=100, activation=tf.nn.softplus)
		self.layer_2 = tf.layers.dense(self.layer_1, units=70, activation=tf.nn.softplus)
		self.layer_3 = tf.layers.dense(self.layer_2, units=20, activation=tf.nn.softplus)
		self.mu = tf.layers.dense(self.layer_3, units=1, activation=None)
		self.var = tf.layers.dense(self.layer_3, units=1, activation=tf.nn.softplus)

		self.likelihood = tfp.distributions.Normal(loc=self.mu, scale=self.var)
		self.out = -self.likelihood.log_prob(y)
		self.mean_loss = tf.reduce_mean(self.out)
		self.mean_y = self.mu + 0.5 * self.var**2.

		self.rmse = tf.sqrt(tf.reduce_mean(y - self.mean_y)**2.)

		self.global_step = tf.Variable(0, trainable=False)
		self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.8).minimize(self.mean_loss,global_step=self.global_step)
		self.init = tf.global_variables_initializer()
		self.init_l = tf.local_variables_initializer()

		# Initialize coefficients
		self.sess = tf.Session()
		self.sess.run(self.init)
		self.sess.run(self.init_l)
