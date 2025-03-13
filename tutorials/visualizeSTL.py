########################################################################################################################
# STL Files Visualization of the whole car
"""

Folder: DrivAerNet_STLs_Combined

This code block illustrates how to visualize 3D STL files using PyVista. The combined STL files are used for aerodynamic
drag prediction, we also provide separate STLs for the front and rear wheels (e.g. for running the CFD simulations).
Please refer to the folder: DrivAerNet_STLs_DoE
"""

import pyvista as pv
import os

# Replace with the actual path to your folder containing .stl files
folder_path = '../3DMeshesSTL'

# List all .stl files in the folder
stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]

# Since we're going for a 2x3 grid, we'll take the first 6 .stl files for visualization
stl_files_to_visualize = stl_files[:6]

# Initialize a PyVista plotter with a 2x3 subplot grid
plotter = pv.Plotter(shape=(2, 3))

# Load and add each mesh to its respective subplot
for i, file_name in enumerate(stl_files_to_visualize):
    # Calculate the subplot position
    row = i // 3  # Integer division determines the row
    col = i % 3  # Modulus determines the column

    # Activate the subplot at the calculated position
    plotter.subplot(row, col)

    # Load the mesh from file
    mesh = pv.read(os.path.join(folder_path, file_name))

    # Add the mesh to the current subplot
    plotter.add_mesh(mesh, color='lightgrey', show_edges=True)

    # Optional: Adjust the camera position or other settings here

# Show the plotter window with all subplots
plotter.show()