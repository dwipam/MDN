import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from network import Network, Network1Comp
import tensorflow as tf
pdf = PdfPages("x.pdf")

# Define Data
x_ = np.arange(-10., 10., .1)
y_ = (np.sin(0.5 * x_) * 5.0 + x_ * 0.5).astype(np.float32)
func = lambda x : np.random.normal(x,x**2./20.,1)
y = []
x = []
for idx, i in enumerate(y_):
	for j in range(20):
		noise = np.random.normal(0,.5,1)
		y+=list(func(i)+noise)
		x.append(x_[idx])

y = np.array(y).reshape((-1,1)).astype(np.float32)
x = np.array(x).reshape((-1,1)).astype(np.float32)

# Read into TF dataset
n=len(x)
EPOCHS = 1000
BATCH_SIZE=n
dataset = tf.data.Dataset \
    					.from_tensor_slices((x, y)) \
    					.repeat(EPOCHS).shuffle(n).batch(BATCH_SIZE)
iter_ = dataset.make_one_shot_iterator()
x_, y_ = iter_.get_next()

network = Network(x_,y_,1)

best_loss = 1e+10
for i in range(EPOCHS * (n//BATCH_SIZE)):
	_, loss, mu, var, pi, x__ = network.sess.run([network.train_op,
													network.mean_loss,
													network.mu, network.var, network.pi,
													network.x])

	if loss < best_loss:
		best_mean_y = mu[:,0]
		best_x = x__
		best_loss = loss
		print("Epoch: {} Loss: {:3.3f}".format(i, loss))

fig, axs = plt.subplots(1,1)
plt.plot(x, y, '.')
plt.plot(best_x, best_mean_y, '.')
pdf.savefig(fig)
plt.close()
pdf.close()
