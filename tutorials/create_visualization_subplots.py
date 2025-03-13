########################################################################################################################
# Advanced Visualization of Mesh and Point Cloud with Pressure Data
"""

Folder: SurfacePressureVTK

This section employs PyVista to conduct an advanced visualization that includes the original 3D mesh,
the mesh with pressure data (surface fields), and a point cloud of the mesh with pressure data.
"""

import pyvista as pv
import numpy as np

def create_visualization_subplots(mesh, pressure_name='p', n_points=100000):
    """
    Create subplots for visualizing the solid mesh, mesh with pressure, and point cloud with pressure.

    Parameters:
    mesh (pyvista.PolyData): The mesh to visualize.
    pressure_name (str): The name of the pressure field in the mesh's point data.
    n_points (int): Number of points to sample for the point cloud.
    """
    camera_position = [(-11.073024242161921, -5.621499358347753, 5.862225824910342),
                       (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
                       (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)]
    # Sample points from the mesh for the point cloud
    if mesh.n_points > n_points:
        indices = np.random.choice(mesh.n_points, n_points, replace=False)
    else:
        indices = np.arange(mesh.n_points)
    sampled_points = mesh.points[indices]
    sampled_pressures = mesh.point_data[pressure_name][indices]

    # Create a point cloud with pressure data
    point_cloud = pv.PolyData(sampled_points)
    point_cloud[pressure_name] = sampled_pressures

    # Set up the plotter
    plotter = pv.Plotter(shape=(1, 3))

    # Solid mesh visualization
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, color='lightgrey')
    plotter.add_text('Solid Mesh', position='upper_left')
    plotter.camera_position = camera_position

    # Mesh with pressure visualization
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars=pressure_name, cmap='jet')
    plotter.add_scalar_bar(title=pressure_name, vertical=True)
    plotter.add_text('Mesh with Pressure', position='upper_left')
    plotter.camera_position = camera_position

    # Point cloud with pressure visualization
    plotter.subplot(0, 2)
    plotter.add_points(point_cloud, scalars=pressure_name, cmap='jet',  clim=(-600, 400), point_size=5)
    plotter.add_scalar_bar(title=pressure_name, vertical=True)
    plotter.add_text('Point Cloud with Pressure', position='upper_left')
    plotter.camera_position = camera_position
    # Show the plot
    plotter.show()

# Load your mesh data here, ensure it has the pressure data in point_data
mesh = pv.read('../SurfacePressureVTK/DrivAer_F_D_WM_WW_3000.vtk')

# Visualize the mesh, mesh with pressure, and point cloud with pressure
create_visualization_subplots(mesh, pressure_name='p', n_points=100000)