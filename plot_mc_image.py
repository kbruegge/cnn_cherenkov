from cnn_cherenkov import image_io
import matplotlib.pyplot as plt
from convert import remap_pixel_values
import numpy as np

X, Y = image_io.load_mc_training_data(N=5000)

gammas = X[Y[:, 0]==1]
protons = X[Y[:, 1] == 1]

fig, [ax1, ax2] = plt.subplots(1, 2)

mask = remap_pixel_values(np.ones(1440)).astype(bool)

img_gamma = np.ma.masked_array(gammas[0, :, :, 0], mask=~mask)
img_proton = np.ma.masked_array(protons[0, :, :, 0], mask=~mask)

ax1.imshow(img_gamma)
ax2.imshow(img_proton)
plt.show()
