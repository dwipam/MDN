import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from network import Network
import tensorflow as tf
pdf = PdfPages("x.pdf")
n=5000

y = np.random.uniform(0., 1., (n,1)).astype(np.float32)
x = y + 0.3 * np.sin(2 * np.pi * y) + np.random.uniform(-0.1, 0.1, size=(n,1)).astype(np.float32)

fig, axs = plt.subplots(1,1)
plt.plot(x,y,'.')
pdf.savefig(fig)
pdf.close()

dataset = tf.data.Dataset \
    					.from_tensor_slices((x, y)) \
    					.shuffle(n).repeat().batch(n)
iter_ = dataset.make_one_shot_iterator()
x_, y_ = iter_.get_next()
network = Network(x_,y_,1)

EPOCHS = 100000
best_loss = 1e+10
for i in range(EPOCHS):
	_, loss, lr = network.sess.run([network.train_op, network.mean_loss, network.learning_rate])
	if loss < best_loss:
		best_loss = loss
		print "Epoch: {} Loss: {:3.3f} LR: {}".format(i, loss, lr)