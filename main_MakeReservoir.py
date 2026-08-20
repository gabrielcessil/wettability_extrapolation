import os
import numpy as np
volume_shape        = (300,300,300)

gt_file             = "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/volume_withAngles.raw"
gt_volume           = np.fromfile(gt_file, dtype=np.uint8).reshape(volume_shape)
print(np.unique(gt_volume), gt_volume.shape)

gt_volume[0:5,:,:]  = 1
gt_volume[-5:,:,:]  = 1
gt_volume.astype(np.uint8).tofile(gt_file)

int_file             = "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/result.raw"
int_volume           = np.fromfile(int_file, dtype=np.uint8).reshape(volume_shape)
print(np.unique(int_volume), int_volume.shape)
int_volume[0:5,:,:]  = 1
int_volume[-5:,:,:]  = 1
int_volume.astype(np.uint8).tofile(int_file)

import Plotter as pl
pl.Plot_Classified_Domain_2D(gt_volume[:,0,:], f"teste_gt", show_label=False,
    special_colors= {
    0: (0.5, 0.5, 0.5, 1),  # Assign grey for solid cells
    1: (0.1, 0.1, 0.1, 1),  # Assign red for fluid I
    2: (0.1, 0.2, 0.6, 1),  # Assign blue for fluid II
    3: (1,   0.549, 0.0, 1)
})
pl.Plot_Classified_Domain_2D(int_volume[:,0,:], f"teste_int", show_label=False,
    special_colors= {
    0: (0.5, 0.5, 0.5, 1),  # Assign grey for solid cells
    1: (0.1, 0.1, 0.1, 1),  # Assign red for fluid I
    2: (0.1, 0.2, 0.6, 1),  # Assign blue for fluid II
    3: (1,   0.549, 0.0, 1)
})