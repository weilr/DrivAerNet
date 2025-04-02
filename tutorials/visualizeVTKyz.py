########################################################################################################################
# Visualization of Pressure and Velocity Slices in yz-plane
"""

Folder: xNormal

In this part, separate PyVista plots are created to visualize pressure ('p') and velocity ('U') slices from a VTK file.
"""
import pyvista as pv

# Replace this with the actual path to your VTK file containing both 'p' and 'U' data
vtk_file_path = 'Z:/DrivAerNet/DrivAerNet++/CFDVTK/E_S_WW_WM_001.vtk'

# Load the VTK file
mesh = pv.read(vtk_file_path)

# Define the final camera position for a horizontal view
final_camera_position = [(20, 0, 0),
                         (4, 0, 3),
                         (0, 0, 1)]

# Plot for 'p' (pressure)
plotter_p = pv.Plotter()  # Create a new plotter instance for 'p'
p_actor = plotter_p.add_mesh(mesh, scalars='p', cmap='jet', clim=(-600, 400), show_scalar_bar=True)
plotter_p.add_text("Pressure (p)", position='upper_left', font_size=20, color='black')
plotter_p.camera_position = final_camera_position  # Set the final camera position
plotter_p.show()  # Display the plot for 'p'

# Plot for 'U' (velocity) using the 'turbo' colormap
plotter_u = pv.Plotter()  # Create a new plotter instance for 'U'
u_actor = plotter_u.add_mesh(mesh, scalars='U', cmap='turbo', clim=(0, 30), show_scalar_bar=True)
plotter_u.add_text("Velocity (U)", position='upper_left', font_size=20, color='black')
plotter_u.camera_position = final_camera_position  # Set the final camera position
plotter_u.show()  # Display the plot for 'U'
