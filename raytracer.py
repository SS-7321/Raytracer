# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:48:15 2022

Design of simulation optical rays using OOP

@author: Samiuzzaman Shan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

global ray_list
ray_list = []
global _exec_order
_exec_order = []
global limits
limits = []


def reset():
    """
    Cleans the environmnt of all rays and objects and closes all figures

    Returns
    -------
    None.

    """
    ray_list.clear()
    _exec_order.clear()
    limits.clear()
    plt.clf()
    plt.cla()
    plt.close()


def optimise_curvatures(focal_length, n_in, n_out, seperation=5,
                        show_dgrm=False, initial_vals=[0.0, -0.0],
                        give_rms=False):
    """
    Optimises the curvtures of a singlet to return two curvature radii that
    would give the smallest RMS at the given focal length from the singlet lens

    Parameters
    ----------
    focal_length : int or float
        The focal length from the lens that the function should optimise
        curvatures for.
    n_in : int or float
        Refractive index of the singlet.
    n_out : int or float
        Refractive index of the medium outside the singlet.
    seperation : int or float, optional
        The largest seperation distance between the two surfaces of the siglet.
        The default is 5.
    show_dgrm : boolean, optional
        Set to True to show the ray diagram and spot diagram of a 10 unit
        radius beam travelling to the focal point. The default is False.
    initial_vals: float or int array of 2 elements
        Initial guesses for the curvatures to pass to scipy.optimize.fmin
    give_rms : boolean, optional
        Set to True to return the RMS of the optimised curvartures. The default
        is False

    Returns
    -------
    Float array
        Array of the curvatures in the following form
    [first lens, second lens].

    """
    rms = 0

    def _singlet_rms(curvatures):
        nonlocal rms
        reset()
        c1 = curvatures[0]
        c2 = curvatures[1]
        make_singlet_lens(c1=c1, c2=c2, n_in=n_in, n_out=n_out, a_r=12, z=0)
        make_beam(centre=[0, 0, -10], radius=10, count=4)
        screen = OutputPlane(seperation + focal_length)
        run()
        rms = screen.rms()
        return rms

    optimised_curvatures = spo.fmin(_singlet_rms, initial_vals)

    if show_dgrm:
        show()
        reset()
    else:
        reset()

    if give_rms:
        return optimised_curvatures, rms
    return optimised_curvatures


def _add_object(output_plane=None, lens=None, rays=None):
    """
    Adds OpticalElement object or Ray class object to the
    simulation environement

    Parameters
    ----------
    output_plane : OutputPlane (OpticalElement) object, optional
        Adds an OpticalPlane object to the simulation environemt.
        The defaultis None.
    lens : SphericalRefraction (OpticalElement) object, optional
        Adds a SphericalRefraction object to the simulation environemt.
        The defaultis None.
    rays : Ray class object, optional
        Adds a Ray class object to the simulation environemt.
        The defaultis None.

    Returns
    -------
    None.

    """
    if output_plane is not None:
        _exec_order.append(output_plane)

    if lens is not None:
        _exec_order.append(lens)

    if rays is not None:
        ray_list.append(rays)


def _order_list():
    """
    Sorts the execution order of the objects in the simulation environment

    Returns
    -------
    None.

    """
    is_sorted = False

    while not is_sorted:
        count = 0

        for i in range(0, len(_exec_order)-1):

            if (_exec_order[i+1].z_intercept < _exec_order[i].z_intercept):
                __temp = _exec_order[i]
                _exec_order[i] = _exec_order[i+1]
                _exec_order[i+1] = __temp
                count += 1

        if count == 0:
            is_sorted = True


def make_beam(centre=[0, 0, 0], radius=1, count=7, color='b', lens=None,
              direction=[0, 0, 1]):
    """
    Creates a bundle of uniformly collimated beams of the specified radius.

    Parameters
    ----------
    centre : 3 element float or int array given as [x,y,z], optional
        The centre of the beam. The default is [0, 0, 0].
    radius : int or float, optional
        The radius of the beam this function generates. The default is 1.
    count : int, optional
        Number of rings of rays the function generates within the beam radius.
        The default is 7.
    color : pyplot applicable color value.
    Either fmt color form or hex color, optional
        The colour of the rays generated. The default is 'b' (Blue).
    lens : SphericalRefraction object, optional
        Automatically adjusts the direction to face the centre of the lens.
        The default is None.
    direction : 3 element float or int array given as [x,y,z], optional
        The direction in which the beam points. The default is [0, 0, 1].

    Returns
    -------
    rays : Ray class object array
        Retun an array of Ray class objects that is used in the simulation.

    """
    density = count / radius

    if lens is not None:
        direction = np.array(centre) * -1 + [0, 0, lens.z_intercept]

    rays = [Ray(p=centre, k=direction)]
    for i in range(1, count):
        r = i / density
        t = np.linspace(0, 2*np.pi, num=(int(np.round(2*np.pi))*i+1),
                        endpoint=True)

        for j in t:
            rays.append(Ray((np.array([r*np.cos(j), r*np.sin(j), 0])
                             + centre), k=direction, color=color))

    return rays


def make_cone(centre=[0, 0, 0], radius=1, count=7, color='b', lens=None,
              direction=[0, 0, 1]):
    """
    Generates a cone of ray beams from a singlular point (centre)

    Parameters
    ----------
    centre : 3 element float or int array given as [x,y,z], optional
        The point at which the rays . The default is [0, 0, 0].
    radius : int or float, optional
        The radius of the beam this function generates. The default is 1.
    count : int, optional
        Number of rings of rays the function generates within the beam radius.
        The default is 7.
    color : pyplot applicable color value.
    Either fmt color form or hex color, optional
        The colour of the rays generated. The default is 'b' (Blue).
    lens : SphericalRefraction object, optional
        Automatically adjusts the direction  and radius to face the centre of
        the lens and change the radius of the beas to be the same as the len's
        aperature radius. The default is None.
    direction : 3 element float or int array given as [x,y,z], optional
        The direction in which the beam points. The default is [0, 0, 1].

    Returns
    -------
    rays : list
        List of Ray class objects to be used in the simulation.

    """
    density = count / radius

    if lens is not None:
        direction = np.array(centre) * -1 + [0, 0, lens.z_intercept]
        radius = lens.aperature_radius

    rays = [Ray(p=centre, k=direction)]

    for i in range(1, count):
        r = i / density
        t = np.linspace(0, 2*np.pi, num=(int(np.round(2*np.pi))*i+1),
                        endpoint=True)

        for j in t:
            rays.append(Ray(p=centre, k=(np.array([r*np.cos(j),
                                                   r*np.sin(j), 0])
                                         + direction), color=color))

    return rays


def calculate_d_scale(diameter, focal_length, wavelength=550):
    """
    Calculates the diffraction scale of the given parameters.

    Parameters
    ----------
    diameter : int or float
        Diameter of the beam.
    focal_length : int or float
        focal length os the refracrive surface.
    wavelength : int or float, optional
        Wavelength in nm of the light rays. The default is 550.

    Returns
    -------
    float
        The diffraction scale of the given parameters.

    """

    return wavelength * (focal_length / diameter) * 1e-9


def make_singlet_lens(c1=0, c2=0, a_r=3, n_in=1, n_out=1,
                      seperation=5, z=100):
    """
    Generates two spherical surfaces to make a singlet and
    adjusts the refractive indicies.

    Parameters
    ----------
    c1 : int or float, optional
        Curvature of the first spherical lens. The default is 0.
    c2 : int or float, optional
        Curvature of the second spherical lens. The default is 0.
    a_r : int or float, optional
        Aperature radius of the singlet lens. The default is 3.
    n_in : int or float
        Refractive index of the singlet.
    n_out : int or float
        Refractive index of the medium outside the singlet.
    seperation : int or float, optional
        The lardest seperation distance between the two surfaces of the siglet.
        The default is 5.
    z : int or float, optional
        The z intercept of the first spherical surface. The default is 100.

    Returns
    -------
    list
        List of 2 SphericalRefraction objects.

    """
    lens1 = SphericalRefraction(a_r=a_r, c=c1, n_1=n_out, n_2=n_in, z=z)
    lens2 = SphericalRefraction(a_r=a_r, c=c2, z=z+seperation, n_1=n_in,
                                n_2=n_out)
    return [lens1, lens2]


def find_focus(lens):
    """
    Finds the focal point of a given SphericalRefraction object

    Parameters
    ----------
    lens : SphericalObject (OpticalElement), optional
        The refractive surface of which this function will find the focal point
        for. The default is None.

    Returns
    -------
    float array of 3 elements
        Return 3D cartesian coordiantes of the focal point in the form
        of [x,y,z].

    """
    ray = Ray(p=[0.01, 0, 0])

    if isinstance(lens, list):
        for elem in lens:
            elem.propagate_ray(ray)
    else:
        lens.propagate_ray(ray)

    ray_list.remove(ray)
    constant = (ray.p()[0] / ray.k()[0]) * -1

    if constant < 0:
        print("Principal focal point is not real")

    return ray.p() + (ray.k() * constant)


def calculate_focal_length(n1=None, n2=None, curvature_radius=None, lens=None):
    """
    Calculates the theoretical focal distance from the spherical surface

    Parameters
    ----------
    n1 : float or int
        Refractive index of the medium of the incident ray.
    n2 : float or int
        Refractive index of the medium of the refracted ray.
    curvature_radius : float or int, optional
        Curvature radius of the spherical lens. The default is None.
    lens : SphericalRefraction (OpticalElement), optional
        The lens the function will calculate the focal point for.
        The default is None.

    Raises
    ------
    Exception
        A lens object must be passed, or all of the other parameters must be
        passed.

    Returns
    -------
    focus : float
        The focal distance from the spherical surface.

    """
    if lens is not None:
        n1 = lens.index_n1
        n2 = lens.index_n2
        curvature_radius = lens.curvature_radius

    if n1 is None and n2 is None and curvature_radius() is None:
        raise Exception("Either a lens object must be given or the refractive"
                        " indices and the curvature radius must be given")
    focus = (n2 / (n2 - n1)) * curvature_radius

    if focus < 0:
        print("Principal focal point is not real")

    return focus


def run():
    """
    Refracts all the rays in the environment with the objects in the
    environment

    Returns
    -------
    None.

    """
    _order_list()

    for obj in _exec_order:
        for ray in ray_list:
            obj.propagate_ray(ray)
            if isinstance(obj, OutputPlane):
                obj.add_points(ray)


def show(show_objects=True, show_spotdgrm=True, show_planes=True):
    """
    Shown the ray diagrams and spot diagrams of the environement

    Parameters
    ----------
    show_objects : boolean, optional
        Shows objects in the environment if True. The default is True.
    show_spotdgrm : boolean, optional
        Shows the spot diagram of each OutputPlane in the environment if True.
        The default is True.

    Returns
    -------
    None.

    """
    global limits
    plt.grid(linestyle='--', linewidth=0.5)

    for ray in ray_list:
        if not ray.is_terminated():
            plt.plot(ray.vertices()[::, 2], ray.vertices()[::, 1], '-',
                     color=ray.color)

    if show_objects:
        for obj in _exec_order:
            if isinstance(obj, SphericalRefraction):
                obj.create()
        limits = [plt.xlim(), plt.ylim()]
        if show_planes:
            for obj in _exec_order:
                if isinstance(obj, OutputPlane):
                    obj.create()

    plt.xlabel("z ($mm$)")
    plt.ylabel("y ($mm$)")
    plt.show()

    if show_spotdgrm:
        for obj in _exec_order:
            if isinstance(obj, OutputPlane):
                obj.screen_output()


class Ray:
    """
    Class to make a Ray class object with intial position and direction

    Parameters
    ----------
    p : 1D float array, 3 elements
        initial position of the ray [x, y, z]
    k : 1D float array, 3 elements
        initial direction of the ray [x, y, z]. This is saved as a unit vector.
    color : pyplot applicable color value, either fmt color form or hex color.
        Sets the color of the ray.
    """

    def __init__(self, p=[0, 0, 0], k=[0, 0, 1], color='b'):

        if np.linalg.norm(k) == 0:
            print("Invalid direction arguement"
                  "\nDirection must have magnitude greater than 0")
            raise NotImplementedError()

        if len(p) != 3 or len(k) != 3:
            raise Exception("Both the position and direction need to have"
                            " exactly 3 integer or float values")
        self._p = np.array([p])
        self._k = np.array([k/np.linalg.norm(k)])
        self.terminated_status = False
        self.color = color
        _add_object(rays=self)

    def p(self):
        """

        Returns
        -------
        1D float array, 3 elements
            Gives the last intercept of the ray with an object in
            the environment.

        """
        return self._p[-1]

    def k(self):
        """

        Returns
        -------
        1D float array, 3 elements
            Gives the last intercept of the ray with an object in
            the environment.

        """

        return self._k[-1]

    def append(self, p, k=None):
        """
        Appends new direction and position to the Ray class object

        Parameters
        ----------
        p : 1D float array, 3 elements
            Cartesian coordiantes of the new position.
        k : 1D float array, 3 elements, optional
            3 vector of the new direction of the ray.
            The default is it's previous direction value.

        Returns
        -------
        None.

        """

        if k is None:
            k = self.k()

        self._p = np.append(self._p, [p], axis=0)
        self._k = np.append(self._k, [k/np.linalg.norm(k)], axis=0)

    def vertices(self):
        """
        Return all the intersect position of the Ray class object

        Returns
        -------
        Float array dimension of [n, 3]
            All n intersects of the Ray class object.

        """
        return self._p

    def terminate(self):
        """
        Terminates the ray so it does not appear on any plots or diagrams

        Returns
        -------
        None.

        """

        self.terminated_status = True

    def is_terminated(self):
        """
        Returns the terminated status of the Ray class object

        Returns
        -------
        boolean
            Returns True of the ray is terminated.

        """

        return self.terminated_status
    
    
class RayBundle(Ray):
        """
        Class to make a bundle of rays.
        
        Parameters
        ----------
        rays : 1D array of Ray class objects
        """
        
        def __init__(self, rays):
            if isinstance(rays, list) is False:
                raise NotImplementedError()
                
            self.rays = rays
            self._planes = []


class OpticalElement:
    """
    Property to be inherited by each optical element
    """

    def propagate_ray(self, ray):
        """
        Propagate a ray through the optical element and appends new position
        (intercept) and direction (refracted ray) to the Ray class object

        Parameters
        ----------
        ray : Ray class object
            ray with initial position and direction

        Returns
        ----------
        None.
        """

        if isinstance(self, OutputPlane):
            do_refract = False

        else:
            do_refract = True
        intercept, normal = self.intercept(ray)

        if do_refract:

            if intercept is not None:
                __new_k = self.refract(ray, normal)

                if __new_k is not None:
                    ray.append(intercept, __new_k)
            else:
                ray.terminate()
        else:
            ray.append(intercept, ray.k())

    def plane_intercept(self, ray, plane_position):
        """
        Calculates the intercept of a Ray class object to a plane.

        Parameters
        ----------
        ray : Ray class object
            Ray object that intersects the plane.
        plane_position : float or int
            The z-position of the plane.

        Returns
        -------
        float array
            Cartesian coordinates of the intercept.

        """

        return ray.p() - (ray.k() * ((ray.p()[2] -
                                      plane_position[2]) / ray.k()[2]))


class SphericalRefraction(OpticalElement):
    """
    Class to make a lens that will propogate rays. If both curvature and curvature radius
    are given, the curvature will be used.

    Parameters
    ----------
    a_r : float
        aperature radius
    c : float
        curvature of the surface
    z : float
        intercept of the surface with the z-axis
    n_1 : float
        refractive index of the medium outside the spherical object
    n_2 : float
        refractive index of the spherical object
    c_radius : float
        radius of curvature of the spherical object
    """

    def __init__(self, a_r=1, c=1, z=10, n_1=1, n_2=1, c_radius=None):
        if c is not None:
            if c == 0:
                c_radius = 0
            else:
                c_radius = 1 / c
            pass
        elif c_radius is not None:
            c = 1/c_radius
        

        self.aperature_radius = a_r
        self.curvature_radius = c_radius
        self.curvature = c
        self.z_intercept = z
        self.index_n1 = n_1
        self.index_n2 = n_2
        self._position = np.array([0, 0, z + c_radius])
        _add_object(lens=self)

    def intercept(self, ray):
        """
        Return the point of intercept and normal unit vctor to the intercept

        Parameters
        ----------
        ray : Ray class object
            ray with initial position and direction
        Returns
        -------
        intercept: 1D float array, 3 elements
            Cartesian coordinates of the intercept
        normal: 1D float array, 3 elements
            3-vector array describing unit normal.

        """
        position_vector = ray.p() - self._position
        dot_product = np.dot(position_vector, ray.k())

        if self.curvature == 0:
            intercept = self.plane_intercept(ray, self._position)
            return self.check_intercept(intercept), np.array([0, 0, -1])

        elif self.curvature > 0:
            constant = -dot_product - np.sqrt(dot_product**2 -
                                              (np.linalg.norm(
                                                  position_vector)**2 -
                                                  (self.curvature_radius)**2))
            intercept = ray.p() + constant * ray.k()
            normal = (intercept - self._position)

        elif self.curvature < 0:
            constant = -dot_product + np.sqrt(dot_product**2 -
                                              (np.linalg.norm(
                                                  position_vector)**2 -
                                                  (self.curvature_radius)**2))
            intercept = ray.p() + constant * ray.k()
            normal = (self._position - intercept)

        if np.isnan(constant) or constant < 0:
            print("No valid intercept")
            return None, None

        normal_hat = normal / np.linalg.norm(normal)

        return self.check_intercept(intercept), normal_hat
    
    def refract(self, ray, normal):
        """
        Refracts the given ray in accordance to Snell's law in vector form in
        3d space

        Parameters
        ----------
        ray : Ray class object
            The ray that is refracted at the boundary of the two mediums.
        normal : 3 element float or int array given as [x,y,z]
            The normal vector at the point of intersection between the ray and
            the boundary. The normal vector must be pointing at the same
            z-direction as the refracted ray.

        Returns
        -------
        refracted_k : 3 element float array given as [x,y,z]
            The direction of the ray after it is refracted at the surface.

        """
        ratio_indecies = self.index_n1 / self.index_n2
        dot_product = np.dot(normal, ray.k())
        theta_outside = np.arccos(dot_product
                                  / (np.linalg.norm(ray.k())
                                     * np.linalg.norm(normal)))

        if np.sin(theta_outside) > ratio_indecies:
            return None

        refracted_k = (ratio_indecies * (ray.k() - normal * dot_product)
                       - np.sqrt(1 - ratio_indecies**2 *
                                 (1 - dot_product**2)) * normal)
        return refracted_k

    def check_intercept(self, intercept):
        """
        Checks to see if the intersect is within the aperature radius of the
        SphericalRefraction object.

        Parameters
        ----------
        intercept : 1D float array, 3 elements
            The intersect of a ray with the SphericalRefraction object.

        Returns
        -------
        intercept : 1D float array, 3 elements or None
            The initial intersect or None if the intersect is not within the
            aperature radius.

        """
        if np.linalg.norm([intercept[0:2]]) > self.aperature_radius:
            print("No valid intercept")
            return None

        return intercept

    def create(self):
        """
        Creates the object in the ray diagram

        Returns
        -------
        None.

        """
        r = abs(self.curvature_radius)

        if self.curvature_radius == 0:
            plt.plot(np.array([1, 1]) * self.z_intercept,
                     np.array([1, -1]) * self.aperature_radius, '-k')

        elif self.curvature_radius > 0:
            lens_angle = np.arctan2(self.aperature_radius, r)
            lens_t = np.linspace(np.pi - lens_angle, np.pi + lens_angle,
                                 num=100, endpoint=True)
            plt.plot(r * np.cos(lens_t)
                     + self.z_intercept + r, r * np.sin(lens_t), 'k')

        elif self.curvature_radius < 0:
            lens_angle = np.arctan2(self.aperature_radius, r)
            lens_t = np.linspace(-lens_angle, lens_angle,
                                 num=100, endpoint=True)
            plt.plot(r * np.cos(lens_t) + self.z_intercept - r,
                     r * np.sin(lens_t), 'k')


class OutputPlane(OpticalElement):
    """
    Class to make the output plane that is parallel to the xy plane

    Parameters
    ----------
    z : float
        intercept of the plane with the z-axis
    """

    def __init__(self, z):

        self.z_intercept = z
        self.position = [0, 0, z]
        _add_object(output_plane=self)
        self._x = []
        self._y = []

    def screen_output(self):
        """
        Gives a plot of the spot radius of the rays intersecting the
        OutputPlane

        Returns
        -------
        None.

        """
        plt.grid()
        plt.scatter(self._x, self._y)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel("x ($mm$)")
        plt.ylabel("y ($mm$)")
        plt.title(f"Spot diagram of plane at z = {self.z_intercept}")
        plt.show()
        self._x

    def add_points(self, ray):
        """
        Adds the intersect of the ray and the OutputPlane

        Parameters
        ----------
        ray : Ray class object
            The ray that intersected with the OutputPlane.

        Returns
        -------
        None.

        """
        if not ray.is_terminated():
            self._x.append(ray.p()[0])
            self._y.append(ray.p()[1])

    def intercept(self, ray):

        return self.plane_intercept(ray, self.position), None

    def get_points(self):
        """
        Returns the x and y coordinates of the spot diagram as seperate array

        Returns
        -------
        float array
            x coordiantes of the spot diagrams.
        float array
            y coordinates of the spot diagram.

        """

        return self._x, self._y

    def rms(self):
        """
        Returns the rms values of the spot diagram

        Returns
        -------
        float
            The RMS of the spot diagrams.

        """
        _r_sq = np.array(np.array(self._x)**2 + np.array(self._y)**2)
        return np.sqrt(np.sum(_r_sq / len(self._x)))

    def create(self):
        """
        Creates the object in the ray diagram

        Returns
        -------
        None.

        """
        ylims = limits[1]
        plt.plot(np.array([1, 1]) * self.z_intercept, np.array(ylims) * 1.1,
                 '-c')
