########################################################################################################################
# Visualization of Pressure and Velocity Slices in xz-plane
"""

Folder: yNormal

In this part, separate PyVista plots are created to visualize pressure ('p') and velocity ('U') slices from a VTK file.
"""

import pyvista as pv

# Replace this with the actual path to your VTK file containing both 'p' and 'U' data
vtk_file_path = '../yNormal/DrivAer_F_D_WM_WW_3000.vtk'

# Load the VTK file
mesh = pv.read(vtk_file_path)

# Settings for a horizontal view
camera_location = (10, -30, 3)
focal_point = (10, 0, 3)
view_up = (0, 0, 1)

# Plot for 'p' (pressure)
plotter_p = pv.Plotter()  # Create a new plotter instance for 'p'
p_actor = plotter_p.add_mesh(mesh, scalars='p', cmap='jet', clim=(-600, 400), show_scalar_bar=True)
plotter_p.add_text("Pressure (p)", position='upper_left', font_size=20, color='black')
plotter_p.camera_position = [camera_location, focal_point, view_up]
plotter_p.show()  # Display the plot for 'p'

# Plot for 'U' (velocity) using the 'turbo' colormap
plotter_u = pv.Plotter()  # Create a new plotter instance for 'U'
u_actor = plotter_u.add_mesh(mesh, scalars='U', cmap='turbo', clim=(0, 30), show_scalar_bar=True)
plotter_u.add_text("Velocity (U)", position='upper_left', font_size=20, color='black')
plotter_u.camera_position = [camera_location, focal_point, view_up]
plotter_u.show()  # Display the plot for 'U'


