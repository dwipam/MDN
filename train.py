import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from network import Network, Network1Comp
import tensorflow as tf
pdf = PdfPages("x.pdf")
n=1000

# Define Data
x = np.random.uniform(.5, 17.5, (n,1)).astype(np.float32)
#y = x + 0.3 * np.sin(2 * np.pi * x) + np.random.uniform(-0.1, 0.1, size=(n,1)).astype(np.float32)
y = np.sin(0.5 * x) * 5.0 + x * 0.5 + np.float32(np.random.normal(size=(n, 1))).astype(np.float32)

# Read into TF dataset
EPOCHS = 2000
BATCH_SIZE=n
dataset = tf.data.Dataset \
    					.from_tensor_slices((x, y)) \
    					.repeat(EPOCHS).shuffle(n).batch(BATCH_SIZE)
iter_ = dataset.make_one_shot_iterator()

all_ = tf.data.Dataset.from_tensor_slices((x, y)).batch(n)
x_, y_ = iter_.get_next()

#network = Network(x_,y_,2)
network1comp = Network1Comp(x_, y_)

best_loss = 1e+10
for i in range(EPOCHS * (n//BATCH_SIZE)):
	# _, loss, lr, mu, var, pi = network.sess.run([network.train_op,
	# 												network.mean_loss,
	# 												network.learning_rate,
	# 												network.mu, network.var, network.pi])
	_2, loss2, rms2, mean_y, x__ = network1comp.sess.run([network1comp.train_op,
													network1comp.mean_loss,
													network1comp.rmse,
													network1comp.mean_y,
													network1comp.x])

	if loss2 < best_loss:
		best_mean_y = mean_y
		best_x = x__
		best_loss = loss2
	# import pdb;pdb.set_trace()
	# print("X: ",x[0])å
	# print("Y: ",y[0])
	# print("Mu: ",mu[0])
	# print("Var: ",var[0])
	# print("Pi: ",pi[0])
	# # pd.DataFrame({'x': x, 'y': y, 'mu': mu, 'sigma': var, 'pi': pi})
	# print("Epoch: {} Loss: {:3.3f} LR: {}".format(i, loss, lr))
	print("Epoch: {} Loss: {:3.3f} RMSE {:3.3f}".format(i, loss2, rms2))
fig, axs = plt.subplots(1,1)
plt.plot(x, y, '.')
plt.plot(best_x, best_mean_y, '.')
pdf.savefig(fig)
plt.close()
pdf.close()
