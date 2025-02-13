### **README.md for `raytracer.py`**

# RayTracer - Optical Ray Tracing Simulation

## Overview
`raytracer.py` is a Python module for simulating the propagation of optical rays through a system of lenses and output planes using **Object-Oriented Programming (OOP)**. The environment automatically manages ray propagation, refraction, and rendering of ray diagrams.

This module is used in **`SimulationTasks.py`**, which demonstrates its functionalities through various optical simulations.

---

## Features
- Simulates **ray refraction and propagation** through optical elements.
- Includes **rays, lenses (spherical refractions), and output planes (screens)**.
- Automatically orders execution based on the **z-intercepts** of objects.
- Supports **ray bundles** such as **collimated beams and conical rays**.
- Provides functions to **find focal points, calculate diffraction scales, and optimize lens curvatures**.
- **Built-in visualization** for ray diagrams and spot diagrams.

---

## Installation & Requirements
### Dependencies
The module requires the following Python libraries:
- **NumPy**
- **SciPy**
- **Matplotlib**

Install dependencies using:
```bash
pip install numpy scipy matplotlib
```

### Running Simulations
Ensure that `raytracer.py` and `SimulationTasks.py` are in the same directory.

Run a simulation using:
```bash
python SimulationTasks.py
```
For **Jupyter Notebook/IPython**, set:
```python
%matplotlib inline
```
to ensure figures render correctly.

---

## Getting Started

### **1. Creating and Running a Simulation**
Basic workflow:
```python
import raytracer as rt

# Create a lens and an output screen
lens = rt.SphericalRefraction(a_r=5, c=0.02, n_1=1, n_2=1.5, z=100)
screen = rt.OutputPlane(z=200)

# Create a bundle of rays
rays = rt.make_beam(count=5, radius=3, lens=lens)

# Run the simulation
rt.run()

# Show the ray diagram
rt.show()
```

### **2. Resetting the Environment**
To clear all objects and prepare for a new simulation:
```python
rt.reset()
```

---

## **Main Components**

### **1. Ray**
The `Ray` class represents a single light ray.
```python
ray = rt.Ray(p=[0, 0, 0], k=[0, 0, 1], color='r')
```
- `p`: Initial position (`[x, y, z]`).
- `k`: Direction vector (`[x, y, z]`).
- `color`: Plot color.

### **2. SphericalRefraction (Lens)**
Represents a **spherical lens surface** for refraction.
```python
lens = rt.SphericalRefraction(a_r=5, c=0.02, n_1=1, n_2=1.5, z=100)
```
- `a_r`: Aperture radius.
- `c`: Curvature.
- `n_1`: Refractive index of surrounding medium.
- `n_2`: Refractive index of lens material.
- `z`: Z-position.

### **3. OutputPlane (Screen)**
Captures rays and generates spot diagrams.
```python
screen = rt.OutputPlane(z=200)
```

### **4. Generating Beams**
- **Collimated Beam:**
  ```python
  ray_list = rt.make_beam(count=7, radius=5, lens=lens)
  ```
- **Conical Beam:**
  ```python
  ray_list = rt.make_cone(centre=[0, 1, 0], lens=lens)
  ```

---

## **Simulation Workflow**
1. **Define Objects** (Lenses, OutputPlanes, and Rays).
2. **Run the Simulation** using:
   ```python
   rt.run()
   ```
3. **View Results:**
   - **Ray Diagram:**
     ```python
     rt.show()
     ```
   - **Spot Diagram:**
     ```python
     screen.screen_output()
     ```

4. **Reset Environment:**
   ```python
   rt.reset()
   ```

---

## **Advanced Features**
### **1. Finding the Focal Point**
```python
focus = rt.find_focus(lens)
print("Focal Point:", focus)
```

### **2. Calculating Focal Length**
```python
focal_length = rt.calculate_focal_length(n1=1, n2=1.5, curvature_radius=50)
print("Focal Length:", focal_length)
```

### **3. Lens Optimization**
Finds the best lens curvatures to minimize **RMS** at a given focal length:
```python
curvatures = rt.optimise_curvatures(focal_length=100, n_in=1.5, n_out=1)
print("Optimized Curvatures:", curvatures)
```

---

## **Using `SimulationTasks.py`**
This script runs various test cases to evaluate `raytracer.py`. 

To execute:
```bash
python SimulationTasks.py
```

### **Example Tasks in `SimulationTasks.py`**
- **Task 9:** Simple test cases for ray refraction.
- **Task 10:** Finding the paraxial focus and comparing it to theoretical calculations.
- **Task 11:** Image projection using a conical beam.
- **Task 12 & 13:** Uniformly collimated beam simulation.
- **Task 14:** Comparing diffraction scale vs RMS.
- **Task 15:** Spot diagram RMS analysis for different beam radii.
- **Lens Optimization:** Finding optimal curvatures to minimize RMS.

---

## **Troubleshooting & Tips**
- **Objects do not appear?**  
  - Ensure `raytracer.run()` is executed before calling `raytracer.show()`.
  - Try calling `raytracer.reset()` before re-running a script.

- **Spot diagrams do not update?**  
  - Call `OutputPlane.get_points()` to extract scatter data.

- **Simulation does not behave correctly?**  
  - Rays must be defined **before** any optical objects.

---

## **Contributors**
Created by **Samiuzzaman Shan**

---

## **License**
This project is open-source. Feel free to modify and use it for research and learning.

```

---

### **Why This README is Structured This Way**
- **Overview** gives a high-level introduction.
- **Features** highlights the key functionalities.
- **Installation & Requirements** ensures the user has all dependencies.
- **Getting Started** provides a **quick-start guide** with sample code.
- **Main Components** explains core classes (`Ray`, `SphericalRefraction`, `OutputPlane`).
- **Simulation Workflow** outlines how to use the module step by step.
- **Advanced Features** showcases more complex calculations (e.g., `find_focus`, `optimise_curvatures`).
- **Using `SimulationTasks.py`** describes pre-built test cases.
- **Troubleshooting & Tips** solves common issues.
- **Contributors & License** acknowledge authorship and rights.

This structure ensures clarity, ease of navigation, and user-friendliness. 🚀