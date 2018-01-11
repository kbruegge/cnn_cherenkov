import matplotlib.pyplot as plt
import numpy as np
import image_io

meta, images = image_io.read_n_rows('./data/crab_images.hdf5',start=0, end=5000 )

import IPython; IPython.embed()
