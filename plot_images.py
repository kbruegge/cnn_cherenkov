from cnn_cherenkov import image_io
import matplotlib.pyplot as plt
import numpy as np

df, images, y = image_io.load_crab_training_data(N=5000)
X, Y = image_io.load_mc_training_data(N=5000)

gammas = X[Y[:, 0]==1]
protons = X[Y[:, 1]==1]

fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(2, 2)
_, b, _ = ax1.hist(np.log10(protons.sum(axis=(1, 2, 3))), bins=100, normed=True, label='protons')
ax1.hist(np.log10(gammas.sum(axis=(1, 2, 3))), bins=b, alpha=0.5, normed=True, label='gammas')
ax1.hist(np.log10(images.sum(axis=(1, 2, 3))), bins=b, alpha=0.5, normed=True, label='data')
ax1.legend()

_, b, _ = ax2.hist((protons.std(axis=(1, 2, 3))), bins=100, normed=True, label='protons')
ax2.hist((gammas.std(axis=(1, 2, 3))), bins=b, alpha=0.5, normed=True, label='gammas')
ax2.hist((images.std(axis=(1, 2, 3))), bins=b, alpha=0.5, normed=True, label='data')
ax2.legend()


ax3.imshow(protons[0, :, :, 0])
ax4.imshow(images[0, :, :, 0])
plt.show()
