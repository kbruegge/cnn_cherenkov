import cnn_cherenkov.image_io as image_io
import matplotlib.pyplot as plt
import numpy as np

df, images = image_io.load_crab_data(0, 10000)
X, Y = image_io.get_mc_training_data(0, 10000)

fig, [ax1, ax2, ax3] = plt.subplots(3, 1)
_, b, _ = ax1.hist(np.log10(X.sum(axis=(1, 2, 3))), bins=100, normed=True)
ax1.hist(np.log10(images.sum(axis=(1, 2, 3))), bins=b, alpha=0.5, normed=True)


ax2.imshow(X[0, :, :, 0])
ax3.imshow(images[0, :, :, 0])
plt.show()
# import IPython; IPython.embed()
