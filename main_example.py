import Plotter as pl
import numpy as np
import utilities as util
import matplotlib.pyplot as plt

morph_drain_file    = "/home/gabriel/Desktop/Molhabilidade/ContactAngle_Interpolation-main --- INTERPORE TEST CODE/Exemplo Medicoes/benthheimer_200x200x200__0_morpho_drain_it1mi.raw"
measurements_file   = "/home/gabriel/Desktop/Molhabilidade/ContactAngle_Interpolation-main --- INTERPORE TEST CODE/Exemplo Medicoes/benthheimer_200x200x200__0_volume_final_morphodrain_AngleMeasures.npy"

volume_shape        = (200,200,200)
morph_drain         = np.fromfile(morph_drain_file, dtype=np.uint8).reshape(volume_shape)
Measurements        = np.load(measurements_file)
morph_drain[(morph_drain != 1) & (morph_drain != 2)]          = 0

angles  = Measurements[0,:]
coord_x = Measurements[1,:].astype(int)
coord_y = Measurements[2,:].astype(int)
coord_z = Measurements[3,:].astype(int)

volume              = morph_drain.copy()
solid_mask          = (morph_drain != 1) & (morph_drain != 2)
volume[solid_mask]  = 0
volume[~solid_mask] = 1

volume_wAngles                            = volume.copy()
volume_wAngles[coord_x, coord_y, coord_z] = angles

sub_morph_drain = morph_drain[100:200,150:200,0:100]

pl.Plot_Classified_Domain(sub_morph_drain, "_morph_drain", 
                          remove_value=[2],
                          special_colors= {
                              0: (0.5, 0.5, 0.5, 1),  # Assign grey for solid cells
                              1: (0.8, 0.36, 0.36, 1),  # Assign red for fluid I
                              2: (0.39, 0.58, 0.93, 1),  # Assign blue for fluid II
                          },
                          
                          )

sub_volume = volume_wAngles[100:200,150:200,0:100]
"""
pl.Plot_Classified_Domain(sub_volume,"_sub_volume_wAngles", 
                          remove_value=[1],
                          special_colors= {
                              0: (0.5, 0.5, 0.5, 0.3), # Assign grey for solid cells
                              1: (0.0, 0.0, 0.0, 1)  # Assign black for void cells (removed from plot)
                          },
                          show_label=False,
                          split_sharp_edges=True
                          )
"""
import numpy as np
import pyvista as pv

def plot_labeled_volume(cell_labels: np.ndarray, label_colors: dict, filename: str):
    """Visualize a 3D labeled volume with custom RGBA colors.
    
    Args:
        cell_labels: 3D numpy array where each value represents a material label
        label_colors: Dictionary mapping labels to RGBA tuples (0-1 range)
        filename: Base filename for output images
    """
    # Create the volume grid
    dimensions = np.array(cell_labels.shape) + 1
    grid = pv.ImageData(dimensions=dimensions.tolist())
    grid.cell_data['labels'] = cell_labels.flatten(order='F')

    # Get all unique labels present in the data
    unique_labels = np.unique(cell_labels)
    max_label = max(unique_labels) if len(unique_labels) > 0 else 0

    # Prepare color and opacity mappings
    colors = []
    opacity = []
    label_list = sorted(label_colors.keys())
    
    for label in label_list:
        r, g, b, a = label_colors[label]
        colors.append([r, g, b, a])
        opacity.append(1)

    # Create plotter
    plotter = pv.Plotter()

    # Create custom colormap (RGB only)
    cmap = pv.LookupTable()
    cmap.values = np.array(colors)

    # Add volume with proper opacity mapping
    plotter.add_volume(
        grid,
        scalars='labels',
        cmap=cmap,
        opacity=dict(zip(label_list, opacity)),  # Map labels to opacities
        blending='composite',
        shade=True,
        scalar_bar_args={
            'title': "Material Types",
            'n_labels': len(label_list),
            'position_x': 0.85,
            'vertical': True,
            'height': 0.4
        }
    )

    # Save outputs
    plotter.show(auto_close=False)
    plotter.screenshot(f"{filename}.png", transparent_background=False)
    plotter.save_graphic(f"{filename}.svg", raster=False, painter=False)
    plotter.close()

def plot_voxel_grid(cell_labels: np.ndarray, label_colors: dict, filename: str):
    """Visualize labeled voxels as discrete cubes with custom colors/transparency."""
    # Create plotter with better lighting
    plotter = pv.Plotter(lighting='three lights')
    
    # Create the structured grid
    dimensions = np.array(cell_labels.shape) + 1
    grid = pv.ImageData(dimensions=dimensions.tolist())
    grid.cell_data['labels'] = cell_labels.flatten(order='F')
    
    # Convert to unstructured grid to access individual cells
    ugrid = grid.cast_to_unstructured_grid()
    
    # Create color array with opacity
    rgba_colors = np.zeros((ugrid.n_cells, 4))  # RGBA array
    
    for label, color in label_colors.items():
        mask = ugrid['labels'] == label
        rgba_colors[mask] = color  # Use the RGBA values directly
    
    # Add scalar array with RGBA values
    ugrid.cell_data['rgba'] = rgba_colors
    #ugrid2 = ugrid.copy()
    
    to_remove_mask = np.argwhere(ugrid["labels"] == 1)
    ugrid.remove_cells(to_remove_mask.flatten(), inplace=True)
    # Add all voxels at once for better performance
    plotter.add_mesh(
        ugrid,
        scalars='rgba',
        rgba=True,  # Indicate we're using RGBA values directly
        show_edges=False,
        edge_color='black',
        line_width=0.1,
        lighting=True,
        style='surface',
        opacity=1
    )
    
    
    plotter.add_mesh(
        ugrid,
        scalars='rgba',
        rgba=True,  # Indicate we're using RGBA values directly
        show_edges=False,
        edge_color='black',
        line_width=0.1,
        lighting=True,
        style='surface',
        opacity=0.5
    )
    
    
    # Configure view
    plotter.show_axes()
    plotter.show_grid()
    plotter.camera_position = 'iso'
    
    # Save outputs
    plotter.show(auto_close=False)
    plotter.screenshot(f"{filename}.png", transparent_background=False)
    plotter.save_graphic(f"{filename}.svg", raster=False)
    plotter.close()
"""   
plot_voxel_grid(sub_morph_drain,
                label_colors= {
                    0: (0.5, 0.5, 0.5, 1),  # Assign grey for solid cells
                    1: (0.8, 0.36, 0.36, 1),  # Assign red for fluid I
                    2: (0.39, 0.58, 0.93, 1),  # Assign blue for fluid II
                },
                filename="_morph_drain1")

plot_labeled_volume(sub_morph_drain,
                    label_colors= {
                        0: (0.5, 0.5, 0.5, 0.1),  # Assign grey for solid cells
                        1: (0.8, 0.36, 0.36, 1),  # Assign red for fluid I
                        2: (0.39, 0.58, 0.93, 1),  # Assign blue for fluid II
                    },
                    filename="_morph_drain")
"""