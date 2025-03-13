########################################################################################################################
# Visualization of full 3D domain
"""

Folder: CFD_VTK

In this part, we visualize the full 3D domain (car and wind tunnel with boundary conditions)
"""

import pyvista as pv

def visualize_flow_field(vtk_file_path, scalar_field='U'):
    """
    Visualize the flow field from a VTK file and allow the user to choose between 'U' (velocity) or 'p' (pressure).

    Parameters:
    vtk_file_path (str): Path to the VTK file containing the flow field data.
    scalar_field (str): Scalar field to visualize ('U' for velocity or 'p' for pressure).

    Returns:
    plotter (pyvista.Plotter): PyVista plotter object with the flow field visualization.
    """
    # Load the VTK file
    mesh = pv.read(vtk_file_path)

    # Create a plotter
    plotter = pv.Plotter()

    # Add the mesh with the selected scalar field
    if scalar_field == 'U':
        plotter.add_mesh(mesh, scalars='U', cmap='turbo', show_scalar_bar=True)
    elif scalar_field == 'p':
        plotter.add_mesh(mesh, scalars='p', cmap='jet', show_scalar_bar=True)
    else:
        raise ValueError("Invalid scalar_field value. Choose either 'U' for velocity or 'p' for pressure.")

    # Set title based on selected scalar field
    if scalar_field == 'U':
        title = "Velocity (U)"
    else:
        title = "Pressure (p)"
    plotter.add_text(title, position='upper_left', font_size=20, color='black')

    return plotter

# Example usage:
vtk_file_path = 'Z:/CFD/N_S_WWS_WM_part1/N_S_WWS_WM_001.vtk'
scalar_field = 'U'  # or 'p' for pressure
plotter = visualize_flow_field(vtk_file_path, scalar_field)
plotter.show()
