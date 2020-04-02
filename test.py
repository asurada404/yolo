import torch

import numpy as np

grid_size = 4
grid = np.arange(grid_size)
a,b = np.meshgrid(grid, grid)
c = a.reshape(-1,1)
d = a.reshape(-1,1)
print(a)

