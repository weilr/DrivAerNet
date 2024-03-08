# DrivAerNet
DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction

## Introduction
DrivAerNet is a large-scale, high-fidelity CFD dataset of 3D industry-standard car shapes designed for data-driven aerodynamic design. It comprises 4000 high-quality 3D car meshes and their corresponding aerodynamic performance coefficients, alongside full 3D flow field information.

## Dataset Details & Contents

The DrivAerNet dataset is meticulously crafted to serve a wide range of applications from aerodynamic analysis to the training of advanced machine learning models for automotive design optimization. It includes:

- **3D Car Meshes**: A total of **4000 designs**, each with **0.5 million elements**, showcasing a variety of conventional car shapes and emphasizing the impact of minor geometric modifications on aerodynamic efficiency.
- **Aerodynamic Coefficients**: Each car model comes with comprehensive **aerodynamic performance coefficients** including drag coefficient (Cd), total lift coefficient (Cl), front lift coefficient (Clf), rear lift coefficient (Clr), and moment coefficient (Cm).
- **CFD Simulation Data**: The dataset features detailed **Computational Fluid Dynamics (CFD) simulation results**, including full 3D pressure, velocity fields, and wall-shear stresses, computed using **8 million mesh elements** for each car design. The total size of this extensive simulation data is around **16TB**.
- **Curated CFD Simulations**: For ease of access and use, a **streamlined version of the CFD simulation data** is provided, refined to include key insights and data, reducing the size to approximately **1TB**. 
- **Storage**: The 3D meshes and aerodynamic coefficients consume about **84GB**, making them manageable for various computational environments.

This rich dataset, with its focus on the nuanced effects of design changes on aerodynamics, provides an invaluable resource for researchers and practitioners in the field.



## Parametric Model 
The DrivAerNet dataset includes a parametric model of the DrivAer fastback, developed using ANSA® software to enable extensive exploration of automotive design variations. This model is defined by 50 geometric parameters, allowing the generation of 4000 unique car designs through Optimal Latin Hypercube sampling and the Enhanced Stochastic Evolutionary Algorithm. 
![DrivAerNetMorphingNew2-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/ed7e825a-db41-4230-ac91-1286c69d61fe)

![ezgif-7-2930b4ea0d](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/f6af36aa-079b-49d9-8ac7-a6b20595faee)


## CFD Data
![Prsentation4-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/3d5e3b3e-4dcd-490f-9936-2a3dbda1402b)

## Car Designs
The DrivAerNet dataset specifically concentrates on conventional car designs, highlighting the significant role that minor geometric modifications play in aerodynamic efficiency. This focus enables researchers and engineers to explore the nuanced relationship between car geometry and aerodynamic performance, facilitating the optimization of vehicle designs for improved efficiency and performance.

https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/86b8046f-8858-4193-a904-f80cc59544d0

## RegDGCNN: Dynamic Graph Convolutional Neural Network for Regression Tasks


## Usage Instructions
The dataset and accompanying Python scripts for data conversion are available at [GitHub repository link].

## Contributing
We welcome contributions to improve the dataset or project. Please submit pull requests for review.

## Maintenance and Support
Maintained by the DeCoDE Lab at MIT. Report issues via [GitHub issues link].

## License
Distributed under the Creative Commons Attribution (CC BY) license. Full terms [here](https://creativecommons.org/licenses/by/4.0/deed.en).

## Citations
Please cite the DrivAerNet dataset in your publications as: [Citation details].

## Additional Resources
- Tutorials: [Link]
- Technical Documentation: [Link]





