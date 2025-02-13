# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:33:48 2022

@author: samiu
"""
# %% Import libraries
import numpy as np
import raytracer as rt
import matplotlib.pyplot as plt
# %% Task 9: Simple test cases
"""
Tests raytracer functions and classes with simple test cases to see of the rays
refract correctly.
"""

ray0 = rt.Ray()
ray1 = rt.Ray(k=[0, 1, 100], color='r')
ray2 = rt.Ray(k=[1, 0, 100], color='r')
ray3 = rt.Ray(k=[0, -1, 100], color='r')
ray4 = rt.Ray(k=[-1, 0, 100], color='r')
ray5 = rt.Ray(p=[0, 1, 0])
ray6 = rt.Ray(p=[0, -1, 0])
ray7 = rt.Ray(p=[1, 0, 0])
ray8 = rt.Ray(p=[-1, 0, 0])

lens = rt.SphericalRefraction(a_r=2, n_1=1, n_2=1.5, z=100, c=0.03)
screen_0 = rt.OutputPlane(450)

rt.run()
plt.title("Ray diagram of simple test rays. Curvature = 0.03 $mm^{-1}$")
rt.show()
rt.reset()


# %% Task 10: Find paraxial focus
"""
Finds the paraxial focus and calculates the percentage difference between the
calculated and found paraxial focus
"""
paraxial_focus = rt.find_focus(lens)[2]
print(f"The experimental paraxial focal point is at z = {paraxial_focus} mm")
calculated_focus = rt.calculate_focal_length(1, 1.5, 1/0.03) + lens.z_intercept
print(f"The calculated paraxial focal point is at z = {calculated_focus} mm")
print(f"The percentage difference between the two focal points are"
      f" {abs(paraxial_focus-calculated_focus)*100/calculated_focus} %")
# %% Task 11: Image projection
"""
Creates a cone of rays that refract with the spherical surface and creates an
image
"""
screen_0 = rt.OutputPlane(500)
screen_1 = rt.OutputPlane(50)
screen_2 = rt.OutputPlane(400)
lens = rt.SphericalRefraction(a_r=2, n_1=1, n_2=1.5, z=100, c=0.03)

ray_list = rt.make_cone(centre=[0, 1, 0], lens=lens)

rt.run()
plt.title("Image projection. Curvature = 0.03 $mm^{-1}$")
rt.show()
rt.reset()


# %% Task 12 and 13: Uniform collimated rays
"""
Creates a uniformly collimated beam and runs the simulation
"""
lens = rt.SphericalRefraction(a_r=7, n_1=1, n_2=1.5, z=100, c=0.03)
ray_list = rt.make_beam(count=7, radius=5, lens=lens)
screen_0 = rt.OutputPlane(rt.find_focus(lens)[2])
screen_1 = rt.OutputPlane(0)

rt.run()
rms = screen_0.rms()
plt.title("Uniformly collimated beam. Curvature = 0.03 $mm^{-1}$")
rt.show()
print(f"RMS value at z = {screen_0.z_intercept} is {screen_0.rms()} mm")
rt.reset()


# %% Task 14: Comparing the diffraction scale with the RMS
"""
Compares the RMS and the diffraction scale of the unformly collimated beam
"""
d_scale = rt.calculate_d_scale(10, 100) * 1e3
print(f"Diffraction scale of lens with curvature 0.03 mm^-1 and ray beam "
      f"of diameter 5 mm is {d_scale} mm.")
print(f"This has a percentage difference of {abs(rms-d_scale)*100/rms} %")


# %% Plano-convex lens ray diagram. Convex surface first
"""
Creates a convex-plano singlet and runs the simulation
"""
n_glass = 1.5168

convex = rt.SphericalRefraction(a_r=10, n_1=1, n_2=n_glass, z=100, c=0.02)
planar = rt.SphericalRefraction(a_r=10, n_1=n_glass, n_2=1, z=105, c=0)
singlet_focus = rt.find_focus([convex, planar])[2]
screen_0 = rt.OutputPlane(singlet_focus)
ray_list = rt.make_beam(radius=10, lens=convex, color='r')
rt.run()
plt.title("Convex-plano singlet. Curvatures = 0.02, 0 $mm^{-1}$")
rt.show()
rt.reset()


# %% Task 15: Plano-convex lens. Spot diagram RMS against beam radius
"""
Generates a graph of RMS against beam radius for a convex-plano singlet
"""
cp_x_beam_radius = []
cp_y_rms = []
data_points = 50

for i in range(1, data_points + 1):
    radius = 10 * i / data_points
    ray_list = rt.make_beam(radius=radius)
    lenses = rt.make_singlet_lens(c1=0.02, c2=0, a_r=12,
                                  n_in=n_glass, n_out=1)
    singlet_focus = rt.find_focus(lenses)[2]
    screen_0 = rt.OutputPlane(singlet_focus)
    rt.run()
    cp_x_beam_radius.append(radius)
    cp_y_rms.append(screen_0.rms())
    rt.reset()

plt.plot(cp_x_beam_radius, cp_y_rms)
plt.title("RMS of the spot diagram against beam radius")
plt.xlabel("Beam radius ($mm$)")
plt.ylabel("RMS of the spot diagram ($mm$)")
plt.grid()
plt.show()


# %% Plano-convex lens ray diagram. Planar surface first
"""
Creates a plano-convex singlet and runs the simulation
"""
screen_0 = rt.OutputPlane(200)
screen_1 = rt.OutputPlane(175)
lenses = rt.make_singlet_lens(c1=0, c2=-0.02, a_r=12, n_in=n_glass, n_out=1)
ray_list = rt.make_beam(radius=10, lens=planar, color='r')
rt.run()
plt.title("Plano-convex singlet. Curvatures = 0, -0.02 $mm^{-1}$")
rt.show()
rt.reset()


# %% Task 15: Plano convex lens. Spot diagram RMS against beam radius
"""
Generates a graph of RMS agains beam radius for a plano-convex singlet
"""
pc_x_beam_radius = []
pc_y_rms = []
data_points = 50

for i in range(1, data_points + 1):
    radius = 10 * i / data_points
    ray_list = rt.make_beam(radius=radius)
    lenses = rt.make_singlet_lens(c1=0, c2=-0.02, a_r=12,
                                  n_in=n_glass, n_out=1, z=100)
    singlet_focus = rt.find_focus(lenses)[2]
    screen_0 = rt.OutputPlane(singlet_focus)
    rt.run()
    pc_x_beam_radius.append(radius)
    pc_y_rms.append(screen_0.rms())
    rt.reset()

plt.plot(pc_x_beam_radius, pc_y_rms)
plt.title("RMS of the spot diagram against beam radius")
plt.xlabel("Beam radius ($mm$)")
plt.ylabel("RMS of the spot diagram ($mm$)")
plt.grid()
plt.show()
rt.reset()

# %% Task 15: Plotting data on same graph
"""
Plots both graphs in the same figure for easier comparison.
"""
plt.plot(cp_x_beam_radius, cp_y_rms, label="Convex-plano singlet")
plt.plot(pc_x_beam_radius, pc_y_rms, label="Plano-convex singlet")
plt.legend()
plt.title("RMS of the spot diagram against beam radius")
plt.xlabel("Beam radius ($mm$)")
plt.ylabel("RMS of the spot diagram ($mm$)")
plt.grid()
plt.show()
rt.reset()


# %% Lens optimisation
"""
Finds the optimal curvatures for the least RMS at f = 100 mm and prints them as
well as the RMS at f = 100 mm of the optimised singlet.
"""
iterations = 4
max_c = (4*5) / (5**2 + 4*12**2)
initial_vals = np.linspace(-max_c, max_c, num=iterations, endpoint=True)
focal_length = 100
n_in = n_glass
n_out = 1
rms_list = []
curvatures_list = []

for i in initial_vals:
    for j in initial_vals:
        guess = np.array([i, j])
        curvatures, rms = rt.optimise_curvatures(focal_length, n_in, n_out,
                                                 initial_vals=guess,
                                                 give_rms=True)
        curvatures_list.append(curvatures)
        rms_list.append(rms)

min_curvatures = curvatures_list[np.argmin(np.array(rms_list))]

print(f"RMS of the optimised singlet's spot diagram = "
      f"{np.array(rms_list).min()} mm")
screen_0 = rt.OutputPlane(100)
lenses = rt.make_singlet_lens(c1=min_curvatures[0], c2=min_curvatures[1],
                              a_r=12, n_in=n_glass, n_out=1, z=-5)
ray_list = rt.make_beam(centre=[0, 0, -10], radius=10, color='g')
rt.run()
plt.title(f"Optimised singlet. Curvatures = {min_curvatures} $mm^{-1}$")
rt.show(show_planes=False)
rt.reset()

# %% Diffraction scale and RMS against beam radius
"""
Finds Diffraction scale and RMS at f = 100 mm for a range of beam radii
starting from 1 to 10. Increments can be adjusted by increasing the value of
data_points.
"""

opt_x_beam_radius = []
opt_y_rms = []
data_points = 50

for i in range(1, data_points + 1):
    radius = 10 * i / data_points
    ray_list = rt.make_beam(radius=radius)
    lenses = rt.make_singlet_lens(c1=min_curvatures[0], c2=min_curvatures[1],
                                  a_r=12, n_in=n_glass, n_out=1, z=10)
    singlet_focus = rt.find_focus(lenses)[2]
    screen_0 = rt.OutputPlane(singlet_focus)
    rt.run()
    opt_x_beam_radius.append(radius)
    opt_y_rms.append(screen_0.rms())
    rt.reset()

# %% Plotting graph
"""
Plots the Diffraction scale and RMS against beam radius
"""
d_scale_y = rt.calculate_d_scale(np.array(opt_x_beam_radius)*2, 100,
                                 wavelength=588,)*1e3
plt.plot(opt_x_beam_radius, opt_y_rms, label="RMS")
plt.plot(opt_x_beam_radius, d_scale_y, label="Diffraction scale")
plt.title("RMS of the spot diagram and Diffraction scale against beam radius")
plt.xlabel("Beam radius ($mm$)")
plt.ylabel("RMS and Diffraction scale ($mm$)")
idx = np.argwhere(np.diff(np.sign(opt_y_rms - d_scale_y))).flatten()[0]
plt.plot(opt_x_beam_radius[idx], opt_y_rms[idx], 'ro', label=f"Beam radius = "
         f"{opt_x_beam_radius[idx]} mm")
plt.legend()
plt.grid()
plt.show()
rt.reset()
print(f"Beam radius of the intersection = {opt_x_beam_radius[idx]} mm")
