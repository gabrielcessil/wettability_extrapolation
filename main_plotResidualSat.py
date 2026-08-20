import Plotter as pl
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import os


gt_filenames = [
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t1500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t2000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t2500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t3000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t3500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t4000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t4500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t5000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t5500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t6000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t6500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t7000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t7500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t8000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t8500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t9000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t9500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t10000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t10500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t11000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t11500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t12000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t12500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t13000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t13500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t14000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t14500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t15000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t15500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t16000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t16500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t17000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t17500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t18000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t18500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t19000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t19500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t20000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t20500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t21000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t21500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t22000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t22500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/id_t23000000.raw",
]

int_filenames = [
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t1500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t2000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t2500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t3000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t3500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t4000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t4500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t5000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t5500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t6000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t6500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t7000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t7500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t8000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t8500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t9000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t9500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t10000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t10500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t11000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t11500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t12000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t12500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t13000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t13500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t14000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t14500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t15000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t15500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t16000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t16500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t17000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t17500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t18000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t18500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t19000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t19500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t20000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t20500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t21000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t21500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t22000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t22500000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape300x300x300__0/Interpolated/id_t23000000.raw",
  
]

volume_shape = (300,300,300)

#"""
# PLOT INVASION DOMAIN STEP-BY-STEP
sat_oil = []
time_steps = []
for i, (gt_filename, int_filename) in enumerate( zip(gt_filenames, int_filenames)):
    print(f"Loading {i}")
    
    gt_volume         = np.fromfile(gt_filename, dtype=np.uint8).reshape(volume_shape)
    it_volume         = np.fromfile(int_filename, dtype=np.uint8).reshape(volume_shape)

    final = it_volume.copy()
    final[ (gt_volume==1) & (it_volume==2) ] = 3
    final[ (gt_volume==2) & (it_volume==1) ] = 4 

    slice_2D = final[:,200,:]
    pl.Plot_Classified_Domain_2D(slice_2D, f"Interpolated_Flooding_{i}", show_label=False,
        special_colors= {
        0: (0.5, 0.5, 0.5, 1),  # Assign grey for solid cells
        1: (0.1, 0.1, 0.1, 1),  # Assign red for fluid I
        2: (0.1, 0.2, 0.6, 1),  # Assign blue for fluid II
        3: (1,   0.549, 0.0, 1), # Assign for where it should be Fluid II (water )but was Fluid I (oil)
        4: (1, 1, 1, 1.0),       # Assign for where it should be Fluid I (oil) but was Fluid II (water)
    })

    print("Should be oil, was oil: ", np.count_nonzero( (gt_volume==1) & (it_volume==1) ))
    print("Should be water, was water: ", np.count_nonzero( (gt_volume==2) & (it_volume==2) ))
    print("Should be water, was oil: ", np.count_nonzero( (gt_volume==2) & (it_volume==1) ))
    print("Should be oil, was water: ", np.count_nonzero( (gt_volume==1) & (it_volume==2) ))
#"""


"""
# PLOT OIL SATURATION COMPARISON CURVE
gt_sat_oil  = []
int_sat_oil = []
time_steps  = []
for i, (gt_file, it_file) in enumerate( zip(gt_filenames, int_filenames) ):
    print(f"Loading {i}")
    gt_volume         = np.fromfile(gt_file, dtype=np.uint8).reshape(volume_shape)
    it_volume         = np.fromfile(it_file, dtype=np.uint8).reshape(volume_shape)

    
    gt_sat_oil.append( np.count_nonzero(gt_volume==1)/np.count_nonzero( (gt_volume==1) | (gt_volume==2) ))
    int_sat_oil.append( np.count_nonzero(it_volume==1)/np.count_nonzero( (it_volume==1) | (it_volume==2) ))
    time_steps.append(i)
  
    
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.family"] = "DejaVu Serif"
fig, ax = plt.subplots(figsize=(19.2, 19.2), dpi=100)
ax.plot(time_steps, gt_sat_oil, marker='^', color='k', linestyle='--',
        markersize=24, linewidth=6, label='Ground Truth domain')
ax.plot(time_steps, int_sat_oil, marker='o', color='k', linestyle='-',
        markersize=28, linewidth=6, label='Extrapolated domain')
ax.set_xlabel('Time steps', fontsize=46)
ax.set_ylabel('Oil Saturation', fontsize=46)
ax.tick_params(axis='both', which='major', labelsize=38)
ax.minorticks_on()
ax.grid(which='both', linestyle='--', alpha=0.5, color='black')
ax.legend(fontsize=34, loc="upper right", frameon=True, shadow=True)
plt.grid(True)
plt.tight_layout()
plt.savefig("Oil_Saturation.svg", dpi=300, bbox_inches="tight")
plt.savefig("Oil_Saturation.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
"""
    