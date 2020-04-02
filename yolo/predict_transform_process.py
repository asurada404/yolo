import torch

import numpy as np

grid_size = 3
grid = np.arange(grid_size)
a, b = np.meshgrid(grid, grid)

x_offset = torch.FloatTensor(a).view(-1, 1)
y_offset = torch.FloatTensor(b).view(-1, 1)

x_y_offset = torch.cat((x_offset, y_offset), 1)
print(x_y_offset.shape)  # (9,2)
print(x_y_offset)

num_anchors = 2
x_y_offset = x_y_offset.repeat(1, num_anchors)
print(x_y_offset.shape)  # (9,4)
print(x_y_offset)

x_y_offset = x_y_offset.view(-1, 2)
print(x_y_offset.shape)  # (18,2)
print(x_y_offset)

x_y_offset = x_y_offset.unsqueeze(0)
print(x_y_offset.shape)  # (1,18,2)  x,y是不是反了??
print(x_y_offset)

anchors = [(1,1),(2,2)]
anchors = torch.FloatTensor(anchors)
anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
print(anchors.shape)