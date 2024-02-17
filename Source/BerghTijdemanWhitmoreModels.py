"""
Normalised Whitmore model, and Bergh and Tijdeman model.

SOURCE of the Bergh & Tijdeman model:
Bergh, H., and Tijdeman, H., “Theoretical and experimental results for the dynamic response
of pressure measuring systems,” Tech. Rep. NLR-TR F. 238, National Aerospace Laboratory NLR,
Amsterdam, The Netherlands, Jan. 1965. http://resolver.tudelft.nl/uuid:e88af84e-120f-4c27-8123-3225c2acd4ad.

SOURCE of the Whitmore model:
Whitmore, S. A., “Frequency response model for branched pneumatic sensing systems,” Journal of Aircraft,
Vol. 43, No. 6, 2006, pp. 1845–1853. https://doi.org/10.2514/1.20759.

ADDITIONAL INTERESTING LINKED PAPER:
- Tijdeman, H., and Bergh, H., “The influence of the main flow on the transfer function of tube-transducer systems
  used for unsteady pressure measurements,” Tech. Rep. NLR MP 72023 U, National Aerospace Laboratory NLR,
  Amsterdam, The Netherlands, Sep. 1972. http://resolver.tudelft.nl/uuid:3d3e7e72-7949-4d1a-8043-3194c42edff5.
- Tijdeman, H., “Investigation of the transonic flow around oscillating airfoils,” Ph.D. thesis,
  Delft Technological University, Delft, The Netherlands, Dec. 1977.
  http://resolver.tudelft.nl/uuid:b07421b9-136d-494c-a161-b188e5ba1d0d
- Tijdeman, H., “On the propagation of sound waves in cylindrical tubes,” Journal of Sound and Vibration,
  Vol. 39, No. 1, 1975, pp. 1–33. https://doi.org/10.1016/S0022-460X(75)80206-9.

- Whitmore, S. A., and Fox, B., “Improved accuracy, second-order response model for pressure
  sensing systems,” Journal of Aircraft, Vol. 46, No. 2, 2009, pp. 491–500. https://doi.org/10.2514/1.36262.
"""
import itertools
import numpy as np
from scipy.special import jv
from typing import Union, Optional, Tuple
import numpy.typing as npt


# --- GENERAL HELPER FUNCTIONS ---
# Convert between dimensionless and dimensional.
def norm_to_dim(k_l_pre: float, alpha_pre: Union[float, complex], vv_vt: float, c0: float = 340.26, nu: float = 1.46E-5,
                alpha_complex: bool = True) -> npt.NDArray[float]:
    """
    Converts non-dimensional tube-cavity geometry parameters (L/c, alpha_pre (real or complex), Vv/Vt), to dimensional
    geometry parameters (tube length, tube radius, cavity volume). Depending on alpha_complex input,
    the input shear wave number constant is either real- or complex-valued.

    :param k_l_pre: Reduced frequency constant k_l_pre = length_i/c0, s.
    :param alpha_pre: Shear wave number constant (complex or real, depending on alpha_complex), s.
        If alpha_complex: alpha_pre = 1j^1.5 * radius_j / sqrt(nu), else: alpha_pre = radius_j / sqrt(nu).
    :param vv_vt: Cavity- to tube-volume ratio, -.
    :param c0: Speed of sound, m/s.
    :param nu: Kinematic viscosity, m^2/s.
    :param alpha_complex: Boolean. If True, then the input shear wave number constant is complex-valued:
        alpha_pre = 1j^1.5 * radius_i / sqrt(nu). If False, then the shear wave number constant is real-valued:
        alpha_pre = radius_i / sqrt(nu).

    :return: Numpy array of dimensional parameters: length_i, radius_i, cavity volume_i.
    """
    length_i = np.real(k_l_pre * c0)  # Tube length.
    if alpha_complex:  # Tube radius.
        radius_i = np.real(alpha_pre * 1j ** -1.5 * nu ** 0.5)
    else:
        radius_i = np.real(alpha_pre * nu ** 0.5)
    volume_i = np.real(vv_vt * np.pi * radius_i ** 2 * length_i)  # Cavity volume.
    return np.array([length_i, radius_i, volume_i])


def dim_to_norm(length_i: float, radius_i: float, volume_i: float, c0: float = 340.26, nu: float = 1.46E-5,
                alpha_complex: bool = True) -> npt.NDArray[Union[float, complex]]:
    """
    Converts dimensional tube-cavity geometry parameters (tube length, tube radius, cavity volume), to non-dimensional
    geometry parameters (L/c, alpha_pre (real or complex), Vv/Vt). Depending on alpha_complex input,
    the shear wave number constant is either real- or complex-valued.

    :param length_i: Tube length, m.
    :param radius_i: Tube radius, m.
    :param volume_i: Cavity volume, m^3.
    :param c0: Speed of sound, m/s.
    :param nu: Kinematic viscosity, m^2/s.
    :param alpha_complex: Boolean. If True, then the shear wave number constant is complex-valued:
        alpha_pre = 1j^1.5 * radius_i / sqrt(nu). If False, then the shear wave number constant is real-valued:
        alpha_pre = radius_i / sqrt(nu).

    :return: Numpy array of normalised parameters: length_i/c0, alpha_pre, Vv/Vt.
    """
    if alpha_complex:  # Shear wave number constant, either complex or real.
        f_alpha = f_alpha_pre_j
    else:
        f_alpha = f_alpha_pre_real_j
    k_l_pre = length_i / c0  # Reduced frequency constant.
    alpha_pre = f_alpha(radius=radius_i, nu=nu)  # Shear wave number constant.
    vv_vt = f_vv_vt(length=length_i, radius=radius_i, volume=volume_i)  # Volume ratio.
    return np.array([k_l_pre, alpha_pre, vv_vt])


# --- WHITMORE, AND BERGH AND TIJDEMAN HELPER FUNCTIONS ---
# Functions used to get the 'normalised' parameters from dimensional parameters.
def j_2_j_0(arg: npt.ArrayLike) -> npt.ArrayLike:
    """
    Fraction of Bessel function of 1st kind of order 2 over order 0.

    :param arg: Argument of Bessel functions of 1st kind.

    :return: Ratio value.
    """
    return jv(2, arg) / jv(0, arg)


def f_alpha_pre_j(radius: float, nu: float = 1.46E-5) -> complex:
    """
    Complex-valued shear wave number pre-multiplier constant used for Bayesian inference.
    alpha_j = alpha_pre_j * omega**0.5. Function computes the alpha_pre_j=1j^1.5*R/sqrt(nu) here.

    :param radius: Tube-radius, m.
    :param nu: Kinematic viscosity, m^2/s.

    :return: Shear wave number complex-valued constant.
    """
    return 1j ** 1.5 * radius * nu ** -0.5


def f_alpha_pre_real_j(radius: float, nu: float = 1.46E-5) -> float:
    """
    Real-valued shear wave number pre-multiplier constant used for Bayesian inference.
    alpha_j = alpha_pre_j * 1j**1.5 * omega**0.5. Function computes the alpha_pre_j=R/sqrt(nu) here.

    :param radius: Tube-radius, m.
    :param nu: Kinematic viscosity, m^2/s.

    :return: Shear wave number real-valued constant.
    """
    return radius * nu**-0.5


def f_vv_vt(length: float, radius: float, volume: float) -> float:
    """
    Cavity- to tube-volume ratio.

    :param length: Tube-length used for tube-volume of final ratio.
    :param radius: Tube-radius used for tube-volume of final ratio.
    :param volume: Cavity-volume used the final ratio.

    :return: Ratio of cavity-volume to tube-volume.
    """
    return volume / (length * np.pi * radius ** 2)


def f_vt_vv(vt_lrv: Tuple[float, float, float], vv_lrv: Tuple[float, float, float]) -> float:
    """
    Simplified function for getting a ratio of tube-volume to cavity-volume of two elements.

    :param vt_lrv: List of tube length, tube radius, and cavity volume for element of which tube-volume is computed.
    :param vv_lrv: List of tube length, tube radius, and cavity volume for element of which cavity-volume is computed.

    :return: Tube-volume w.r.t. cavity-volume.
    """
    return 1/f_vv_vt(length=vt_lrv[0], radius=vt_lrv[1], volume=vv_lrv[2])


# Functions used in Element and ElementBT class, i.e. the underlying models.
def f_p_ratio_j(phi_l_j: npt.NDArray[complex], ve_vt_j: npt.NDArray[Union[float, complex]]):
    """
    Pressure ratio of tube-cavity element j. Pressure at cavity w.r.t. inlet of tube, p_j/p_{j-1}.

    :param phi_l_j: Wave propagation factor time tube-length of element j, -.
    :param ve_vt_j: Effective-volume to tube-volume ratio of element j, -.

    :return: Array for pressure ratio of element j.
    """
    return 1 / (np.cosh(phi_l_j) + ve_vt_j * phi_l_j * np.sinh(phi_l_j))


def f_ve_vt_j(vv_vt_j: float, sum_phi_j: Optional[Union[float, npt.NDArray[complex]]] = 0.) -> \
        npt.NDArray[Union[float, complex]]:
    """
    Effective-volume to tube-volume ratio of element j.

    :param vv_vt_j: Cavity-volume w.r.t. tube-volume ratio, -.
    :param sum_phi_j: Complex impedance of element j, sum of all child elements, -.

    :return: Array of effective- to tube-volume ratio, -.
    """
    return vv_vt_j * (1 + sum_phi_j)


def f_n_j(alpha_j: npt.NDArray[complex], gamma: float, pr: float) -> npt.NDArray[complex]:
    """
    Polytropic exponent of element j, -.

    :param alpha_j: Shear wave number of element j, -.
    :param gamma: Ratio of specific heats, -.
    :param pr: Prandtl number, -.

    :return: Array for polytropic exponent.
    """
    return 1 / (1 + (gamma - 1) / gamma * j_2_j_0(alpha_j * pr ** 0.5))


def f_phi_j(w: npt.NDArray[float], alpha_j: npt.NDArray[complex], gamma: float, pr: float, c0: float) \
        -> npt.NDArray[complex]:
    """
    Wave propagation factor.

    :param w: Angular frequency array, rad/s.
    :param alpha_j: Shear wave number of element j, -. alpha_j = 1j**1.5 * R_j * (omega/nu)**0.5.
    :param gamma: Ratio of specific heats, -.
    :param pr: Prandtl number, -.
    :param c0: Speed of sound, m/s.

    :return: Wave propagation factor array.
    """
    n_j = f_n_j(alpha_j, gamma, pr)  # Polytropic constant.
    return w / c0 * np.sqrt(gamma / (n_j * j_2_j_0(alpha_j)))


def f_phi_l_j(k_l_j: float, alpha_j: npt.NDArray[complex], gamma: float, pr: float) -> npt.NDArray[complex]:
    """
    Wave propagation factor times tube-length.

    :param k_l_j: Reduced frequency, k_L_j = w L_j / c_j [-], where w is the angular frequency array.
    :param alpha_j: Shear wave number of element j, -. alpha_j = 1j**1.5 * R_j * (omega/nu)**0.5.
    :param gamma: Ratio of specific heats, -.
    :param pr: Prandtl number, -.

    :return: Wave propagation factor times tube-length array.
    """
    n_j = f_n_j(alpha_j, gamma, pr)
    return k_l_j * np.sqrt(gamma / (n_j * j_2_j_0(alpha_j)))


def f_phi_ij(vt_i_vv_j: float, c_i_c_j: float, phi_l_i: npt.NDArray[complex],
             ve_vt_i: npt.NDArray[Union[float, complex]]) -> npt.NDArray[complex]:
    """
    Complex impedance of element i w.r.t. base node j.

    :param vt_i_vv_j: Ratio of tube-volume of element w.r.t. cavity-volume of base element.
    :param c_i_c_j: Ratio of speed-of-sound of element w.r.t. base element.
    :param phi_l_i: Wave propagation factor time tube-length of element i.
    :param ve_vt_i: Ratio of effective-volume w.r.t. tube-volume of element i.

    :return: Part of complex impedance of element i for base node j.
    """
    c_pl, s_pl = np.cosh(phi_l_i), np.sinh(phi_l_i)
    pl_ve__vt = phi_l_i * ve_vt_i
    return ve_vt_i * vt_i_vv_j * c_i_c_j ** -2 * (c_pl + s_pl / pl_ve__vt) / (c_pl + s_pl * pl_ve__vt)


# --- BERGH & TIJDEMAN MODEL ---
# Specific probe topologies.
def bt_pinhole(w_arr: npt.NDArray[float], k_l_pre: float, alpha_pre: complex, vv_vt: float, sigma: float = 0.,
               gamma: float = 1.4, k_n: float = 1., pr: float = 0.7,
               p1_p0: Optional[Union[float, npt.NDArray[complex]]] = 0.,
               k_l_pre_1: Optional[float] = 0., alpha_pre_1: Optional[complex] = 0., vt1_vv0: Optional[float] = 0.,
               l1_l0: Optional[float] = 0.) -> npt.NDArray[complex]:
    """
    Bergh & Tijdeman model tube-transducer element with normalised input parameters.
    Models a singular element.

    :param w_arr: Angular frequency array, w = 2 pi f [rad/s].
    :param k_l_pre: Wave-number constant [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre: Shear wave number constant [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt: Cavity-volume to tube-volume ratio [-]. vv_vt = Vv / (L pi R^2).
    :param sigma: Dimensionless factor giving the increase in volume due to diaphragm deflection. Default = 0.
    :param gamma: Specific heat ratio (constant) [-].
    :param k_n: Ratio of polytropic constant in cavity (k) w.r.t. specific heat ratio in the tube (n).
        Default = 1 (k = n).
    :param pr: Prandtl number [-].

    :param p1_p0: Pressure ratio of attached child-element. Default = 0 (no child).
    :param k_l_pre_1: Wave-number constant of child element [s]. If p1_p0 = 0, then not considered.
    :param alpha_pre_1: Shear wave number constant of child element [s^(1/2)]. If p1_p0 = 0, then not considered.
    :param vt1_vv0: Tube-volume of element 1 w.r.t. cavity-volume of element 0 [-].
        Vt_child__Vv_base = (L pi R^2)_1 / (Vv)_0.
    :param l1_l0: Tube-length ratio of child element w.r.t. base element [-].

    :return: Pressure ratio array (for all elements of w_arr) of cavity w.r.t. inlet of tube.
    """
    # Wave-number after multiplying the wave-number constant with the angular frequency array.
    _k_l_j = k_l_pre * w_arr
    # Shear wave number after multiplying the Shear wave number constant with the angular frequency array.
    _alpha_j = alpha_pre * np.sqrt(w_arr)

    _n_j = f_n_j(alpha_j=_alpha_j, gamma=gamma, pr=pr)  # Ratio of specific heats.
    # Wave propagation factor * tube length.
    _phi_l_j = f_phi_l_j(k_l_j=_k_l_j, alpha_j=_alpha_j, gamma=gamma, pr=pr)
    _k_j = k_n * _n_j  # Polytropic constant in cavity.
    _ve_vt_j = vv_vt * (sigma + 1 / _k_j) * _n_j  # Effective volume over tube volume, Ve/Vt [-].

    if p1_p0 != 0:  # Added effect of child elements to effective-volume to tube-volume ratio of element.
        _k_l_1 = k_l_pre_1 * w_arr
        _alpha_1 = alpha_pre_1 * np.sqrt(w_arr)
        _phi_l_1 = f_phi_l_j(k_l_j=_k_l_1, alpha_j=_alpha_1, gamma=gamma, pr=pr)
        _ve_vt_j_term2 = vt1_vv0*vv_vt * _phi_l_1 * (_phi_l_j*l1_l0) ** -2 * j_2_j_0(_alpha_1)/j_2_j_0(_alpha_j) * \
                         (np.cosh(_phi_l_j) - p1_p0) / np.sinh(_phi_l_1)
        _ve_vt_j += _ve_vt_j_term2
    return f_p_ratio_j(phi_l_j=_phi_l_j, ve_vt_j=_ve_vt_j)  # Pressure ratio of cavity w.r.t. inlet of tube.


def bt_series(w_arr: npt.NDArray[float], k_l_pre_0: float, alpha_pre_0: complex, vv_vt_0: float,
              k_l_pre_1: float, alpha_pre_1: complex, vv_vt_1: float, vt1_vv0: float, l1_l0: float, sigma_0: float = 0.,
              k_n0: float = 1., sigma_1: float = 0., k_n1: float = 1., gamma: float = 1.4, pr: float = 0.7,
              p2_p1: Optional[Union[float, npt.NDArray[complex]]] = 0., k_l_pre_2: Optional[float] = 0.,
              alpha_pre_2: Optional[complex] = 0., vt2_vv1: Optional[float] = 0., l2_l1: Optional[float] = 0.) -> \
        Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
    """
    Bergh & Tijdeman model tube-transducer element with normalised input parameters.
    Models two singular elements in series. Element 0 at inlet port. Element 1 has tube coupled to cavity of element 0.

    :param w_arr: Angular frequency array, w = 2 pi f [rad/s].
    :param k_l_pre_0: Wave-number constant of element 0 [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre_0: Shear wave number constant of element 0 [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt_0: Cavity-volume to tube-volume ratio of element 0 [-]. vv_vt = Vv / (L pi R^2).
    :param k_l_pre_1: Wave-number constant of element 1 [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre_1: Shear wave number constant of element 1 [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt_1: Cavity-volume to tube-volume ratio of element 1 [-]. vv_vt = Vv / (L pi R^2).
    :param vt1_vv0: Tube-volume of element 1 w.r.t. cavity-volume of element 0 [-].
            Vt_child__Vv_base = (L pi R^2)_1 / (Vv)_0.
    :param l1_l0: Tube-length ratio of child element w.r.t. base element [-].
    :param sigma_0: Dimensionless factor giving the increase in volume due to diaphragm deflection of element 0.
        Default = 0.
    :param k_n0: Ratio of polytropic constant in cavity (k) w.r.t. specific heat ratio in the tube (n) of element 0.
        Default = 1 (k = n).
    :param sigma_1: Dimensionless factor giving the increase in volume due to diaphragm deflection of element 1.
        Default = 0.
    :param k_n1: Ratio of polytropic constant in cavity (k) w.r.t. specific heat ratio in the tube (n) of element 1.
        Default = 1 (k = n).
    :param gamma: Specific heat ratio (constant) [-].
    :param pr: Prandtl number [-].

    :param p2_p1: Pressure ratio of child-element attached to element 1. Default = 0 (no child).
    :param k_l_pre_2: Wave-number constant of child-element attached to element 1 [s].
        If p2_p1 = 0, then not considered.
    :param alpha_pre_2: Shear wave number constant of child element [s^(1/2)]. If p2_p1 = 0, then not considered.
    :param vt2_vv1: Tube-volume of child element of element 1 w.r.t. cavity-volume of element 1 [-].
        Vt_child__Vv_base = (L pi R^2)_1 / (Vv)_0.
    :param l2_l1: Tube-length ratio of child element of element 1 w.r.t. element 1 [-].

    :return: Tuple containing: Array of pressure ratio of element 0, array of pressure ratio of element 1.
    """
    # Start from closed-off end of the tube. Compute the pressure ratio of these elements,
    # use in base elements iteratively.
    pr1 = bt_pinhole(w_arr=w_arr, k_l_pre=k_l_pre_1, alpha_pre=alpha_pre_1, vv_vt=vv_vt_1, sigma=sigma_1, gamma=gamma,
                     k_n=k_n1, pr=pr, p1_p0=p2_p1, k_l_pre_1=k_l_pre_2, alpha_pre_1=alpha_pre_2, vt1_vv0=vt2_vv1,
                     l1_l0=l2_l1)

    pr0 = bt_pinhole(w_arr=w_arr, k_l_pre=k_l_pre_0, alpha_pre=alpha_pre_0, vv_vt=vv_vt_0, sigma=sigma_0, gamma=gamma,
                     k_n=k_n0, pr=pr, p1_p0=pr1, k_l_pre_1=k_l_pre_1, alpha_pre_1=alpha_pre_1, vt1_vv0=vt1_vv0,
                     l1_l0=l1_l0)
    return pr0, pr1


# Class used to build any type of serial probe topology more easily.
class ElementBT:
    # Used to assign a unique ID to the object when it is created.
    id_iter = itertools.count()

    def __init__(self, k_l_pre: float, alpha_pre: complex, vv_vt: float, sigma: float = 0., k_n: float = 1.,
                 gamma: float = 1.4, pr: float = 0.7, id_name: Optional[str] = None):
        """
        Bergh & Tijdeman model tube-transducer element with normalised input parameters.
        Can model a singular element, or elements in series.

        Modified to resemble normalised Whitmore model, allows one to use the same functions,
        and compare how the models may result in different TFs.

            Start building probe topology starting from elements furthest from the probe inlet.
            Assign the furthest element to closer elements as children with the add_child method,
            until one gets to the probe inlet element.
            Then compute the pressure ratio of each element w.r.t. its base element using the p_ratio method.
            Do this for all the elements that make up the path of the desired TF,
            and multiply the pressure ratios together accordingly.

        :param k_l_pre: Wave-number constant [s], i.e. L/c. k_L = k_l_pre * omega.
        :param alpha_pre: Shear wave number constant [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
            alpha = alpha_pre * omega^(1/2).
        :param vv_vt: Cavity-volume to tube-volume ratio [-]. vv_vt = Vv / (L pi R^2).
        :param sigma: Dimensionless factor giving the increase in volume due to diaphragm deflection. Default = 0.
        :param k_n: Ratio of polytropic constant in cavity (k) w.r.t. specific heat ratio in the tube (n).
            Default = 1 (k = n).
        :param gamma: Specific heat ratio [-].
        :param pr: Prandtl number [-].
        :param id_name: Name used in representation, set as {id} property. Default None: ID integer given to object.
        """
        # Element parameters.
        self.k_l_pre = k_l_pre
        self.alpha_pre = alpha_pre
        self.sigma = sigma
        self.k_n = k_n
        self.vv_vt = vv_vt
        self.gamma = gamma
        self.pr = pr
        # Element ID.
        if id_name is None:
            self.id = next(ElementBT.id_iter)
        else:
            self.id = id_name

        # Coupling of elements. Tree structure.
        self.base = None
        self.child = None
        # Parameter ratio of coupled elements.
        self.vt_child__vv_elem = None
        self.c_child__c_elem = None
        self.l_child__l_elem = None

        # Not directly shown variables for representation etc.
        self.repr_format = 'E{self.id}'
        self.str_format = '(Element:{self.__repr__()}|Base:{self.base.__repr__()}|Child:{self.child})'

    def __repr__(self) -> str:
        """
        Representation of Bergh & Tijdeman model tube-transducer element with normalised input parameters.

        :return: String of representation.
        """
        return f'E{self.id}'

    def __str__(self) -> str:
        """
        String print of Bergh & Tijdeman model tube-transducer element with normalised input parameters.

        :return: String of printing format.
        """
        return f'(Element:{repr(self)}|Base:{repr(self.base)}|Child:{self.child})'

    def __call__(self, w: npt.NDArray[float]) -> npt.NDArray[complex]:
        """
        Calls the p_ratio method. ! Only return the pressure ratio !

        Compute complex pressure ratio of the node at the cavity to the tube-inlet,
        i.e. the pressure ratio at the transducer cavity w.r.t. the base element of this element, p_j / p_{j-1} [-].
        Considers the effect of child-elements.

        :param w: Angular frequency, [rad/s]. w = 2 pi f.

        :return: Array of pressure ratio.
        """
        return self.p_ratio(w=w)[0]

    def add_child(self, child_element: 'ElementBT', vt_child__vv_elem: float,
                  l_child__l_elem: float, c_child__c_elem: float = 1.):
        """
        Add child element to this element. Element set as base to children. Children given parameter values of input.

        :param child_element: Whitmore model tube-transducer element with normalised input parameters
            appended as child to this node.
        :param vt_child__vv_elem: Child-node-tube-volume to base-node-cavity-volume ratio [-].
            Vt_child__Vv_base = (L pi R^2)_child / (Vv)_base.
        :param l_child__l_elem: Ratio of child-to-base tube-length [-].
        :param c_child__c_elem: Ratio of child-to-base speed of sound [-]. Default: 1.0.

        :return: None.
        """
        child_element.base = self  # For the child, set the base element.
        self.vt_child__vv_elem = vt_child__vv_elem  # For the child, set the volume-ratio.
        self.c_child__c_elem = c_child__c_elem  # For the child, set the speed of sound ratio.
        self.l_child__l_elem = l_child__l_elem  # For the child, set the tube length ratio.
        self.child = child_element  # For this base element, set the child.

    def p_ratio(self, w: npt.NDArray[float]) -> Tuple[npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]]:
        """
        Compute complex pressure ratio of the node at the cavity to the tube-inlet,
        i.e. the pressure ratio at the transducer cavity w.r.t. the base element of this element, p_j / p_{j-1} [-].
        Considers the effect of child-elements.

        :param w: Angular frequency, [rad/s]. w = 2 pi f.

        :return: Array of pressure ratio, Array of shear wave number of element,
            Array of wave propagation constant times tube length.
        """
        _k_l_j = self.k_l_pre * w
        _alpha_j = self.alpha_pre * np.sqrt(w)
        _n_j = f_n_j(alpha_j=_alpha_j, gamma=self.gamma, pr=self.pr)
        _phi_l_j = f_phi_l_j(k_l_j=_k_l_j, alpha_j=_alpha_j, gamma=self.gamma, pr=self.pr)
        _k_j = self.k_n * _n_j
        _ve_vt_j = self.vv_vt * (self.sigma + 1 / _k_j) * _n_j

        # Child
        # If the object has a child element, then add contribution to Ve/Vt.
        if self.child is not None:
            p_child__p_base, _alpha_j1, _phi_l_j1 = self.child.p_ratio(w=w)
            _ve_vt_j_term2 = self.vt_child__vv_elem*self.vv_vt*_phi_l_j1 * (_phi_l_j * self.l_child__l_elem)**-2 * \
                             j_2_j_0(_alpha_j1)/j_2_j_0(_alpha_j)*(np.cosh(_phi_l_j)-p_child__p_base)/np.sinh(_phi_l_j1)
            _ve_vt_j += _ve_vt_j_term2

        p_elem__p_base = f_p_ratio_j(phi_l_j=_phi_l_j, ve_vt_j=_ve_vt_j)
        return p_elem__p_base, _alpha_j, _phi_l_j


def bt_element_example():
    """
    Example of using ElementBT object.
    """
    # Define parameters.
    lrv_a, lrv_b = (2E-3, 0.5E-3 / 2, 20E-9), (6E-3, 0.4E-3 / 2, 40E-9)
    f_arr = np.logspace(np.log10(1E2), np.log10(1E4), 1000)
    w_arr = 2*np.pi*f_arr
    # Volume ratios for child-parent objects.
    l_b__l_a, vt_b__vv_a = lrv_b[0] / lrv_a[0], lrv_b[0] * np.pi * lrv_b[1] ** 2 / lrv_a[2]
    # Define objects.
    element_a = ElementBT(*dim_to_norm(*lrv_a, alpha_complex=True))
    element_b = ElementBT(*dim_to_norm(*lrv_b, alpha_complex=True))
    # Assign children, in correct order.
    element_a.add_child(child_element=element_b, vt_child__vv_elem=vt_b__vv_a,
                        l_child__l_elem=l_b__l_a, c_child__c_elem=1.)
    # Compute desired pressure ratios.
    pr_a, pr_b = element_a(w=w_arr), element_b(w=w_arr)
    pr_total = pr_a * pr_b

    import matplotlib.pyplot as plt
    from Source.PlottingFunctions import pi_scale
    from Source.ProcessingFunctions import frequency_response
    amp_a, phase_a = frequency_response(pr_a)
    amp_b, phase_b = frequency_response(pr_b)
    amp_tot, phase_tot = frequency_response(pr_total)
    fig_tf, ax_tf = plt.subplots(2, 1, sharex='col')
    str_0, str_1, str_tot = r"$p'_1/p'_0$", r"$p'_2/p'_1$", r"$p'_2/p'_0$"
    # Element 0
    ax_tf[0].plot(f_arr, amp_a, color='b', linestyle='--', label=str_0)
    ax_tf[1].plot(f_arr, phase_a, color='b', linestyle='--', label=str_0)
    # Element 1
    ax_tf[0].plot(f_arr, amp_b, color='r', linestyle='-.', label=str_1)
    ax_tf[1].plot(f_arr, phase_b, color='r', linestyle='-.', label=str_1)
    # Total
    ax_tf[0].plot(f_arr, amp_tot, color='k', linestyle='-', label=str_tot)
    ax_tf[1].plot(f_arr, phase_tot, color='k', linestyle='-', label=str_tot)
    # Axis limits.
    ax_tf[0].set_ylim(2E-3, 4)
    pi_scale(min_val=-2*3.15, max_val=0.1, ax=ax_tf[1], pi_minor_spacing=0.5)
    ax_tf[1].set_xlim(f_arr[0], f_arr[-1])
    # Plotting style.
    ax_tf[0].set_yscale('log')
    ax_tf[0].set_ylabel('|TF|, -')
    ax_tf[1].set_ylabel(r'$\angle$TF, rad')
    ax_tf[1].set_xlabel('f, Hz')
    ax_tf[1].set_xscale('log')
    ax_tf[0].grid(which='both')
    ax_tf[1].grid(which='both')
    ax_tf[1].legend(loc='lower left')
    fig_tf.show()


# --- WHITMORE MODEL ---
# Specific Topologies.
def whitmore_pinhole(w_arr: npt.NDArray[float], k_l_pre: float, alpha_pre: complex, vv_vt: float, gamma: float = 1.4,
                     pr: float = 0.7, sum_i_phi_ij: Optional[Union[float, npt.NDArray[complex]]] = 0.) -> \
        npt.NDArray[complex]:
    """
    Whitmore model tube-transducer element with normalised input parameters. Models a singular element.

    :param w_arr: Angular frequency array, w = 2 pi f [rad/s].
    :param k_l_pre: Wave-number constant [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre: Shear wave number constant [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt: Cavity-volume to tube-volume ratio [-]. vv_vt = Vv / (L pi R^2).
    :param gamma: Specific heat ratio [-].
    :param pr: Prandtl number [-].
    :param sum_i_phi_ij: Complex impedance of all child elements to this element. Default = 0 (no child elements).

    :return: Pressure ratio array (for all elements of w_arr) of cavity w.r.t. inlet of tube.
    """
    # Wave-number after multiplying the wave-number constant with the angular frequency array.
    _k_l_j = k_l_pre * w_arr
    # Shear wave number after multiplying the Shear wave number constant with the angular frequency array.
    _alpha_j = alpha_pre * np.sqrt(w_arr)

    # Wave propagation factor (phi) * tube length (L), [-].
    _phi_l_j = f_phi_l_j(k_l_j=_k_l_j, alpha_j=_alpha_j, gamma=gamma, pr=pr)
    # Effective volume over tube volume, Ve/Vt [-].
    _ve_vt_j = f_ve_vt_j(vv_vt_j=vv_vt, sum_phi_j=sum_i_phi_ij)
    # Pressure ratio of cavity w.r.t. inlet of tube.
    return f_p_ratio_j(phi_l_j=_phi_l_j, ve_vt_j=_ve_vt_j)


def whitmore_element_complex_impedance(w_arr: npt.NDArray[float], k_l_pre: float, alpha_pre: complex, vv_vt: float,
                                       vt_elem__vv_base: float, c_elem__c_base: float = 1., gamma: float = 1.4,
                                       pr: float = 0.7,
                                       sum_i_phi_ij: Optional[Union[float, npt.NDArray[complex]]] = 0.) -> \
        npt.NDArray[complex]:
    """
    Complex impedance of the element w.r.t. a base element, Phi_ij [-].

    :param w_arr: Angular frequency, [rad/s]. w = 2 pi f.
    :param k_l_pre: Wave-number constant [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre: Shear wave number constant [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt: Cavity-volume to tube-volume ratio [-]. vv_vt = Vv / (L pi R^2).
    :param vt_elem__vv_base: Child-node-tube-volume to base-node-cavity-volume ratio [-].
            Vt_child__Vv_base = (L pi R^2)_child / (Vv)_base.
    :param c_elem__c_base: Ratio of child-to-base speed of sound [-]. Default: 1.0.
    :param gamma: Specific heat ratio [-].
    :param pr: Prandtl number [-].
    :param sum_i_phi_ij: Complex impedance of all child elements to this element. Default = 0 (no child elements).

    :return: Array of complex impedance.
    """
    # i: elem | j: base | k: elements that use element i as base element
    # Wave-number after multiplying the wave-number constant with the angular frequency array.
    _k_l_i = k_l_pre * w_arr
    # Shear wave number after multiplying the Shear wave number constant with the angular frequency array.
    _alpha_i = alpha_pre * np.sqrt(w_arr)

    # Wave propagation factor (phi) * tube length (L), [-].
    _phi_l_i = f_phi_l_j(k_l_j=_k_l_i, alpha_j=_alpha_i, gamma=gamma, pr=pr)
    # Effective volume over tube volume, Ve/Vt [-].
    _ve_vt_i = f_ve_vt_j(vv_vt_j=vv_vt, sum_phi_j=sum_i_phi_ij)

    complex_impedance_j = f_phi_ij(vt_i_vv_j=vt_elem__vv_base, c_i_c_j=c_elem__c_base,
                                   phi_l_i=_phi_l_i, ve_vt_i=_ve_vt_i)
    return complex_impedance_j


def whitmore_series(w_arr: npt.NDArray[float], k_l_pre_0: float, alpha_pre_0: complex, vv_vt_0: float, k_l_pre_1: float,
                    alpha_pre_1: complex, vv_vt_1: float, vt1_vv0: float, c1_c0: float = 1., gamma: float = 1.4,
                    pr: float = 0.7, sum_2_phi_ij: Optional[Union[float, npt.NDArray[complex]]] = 0.) -> \
        Tuple[npt.NDArray[complex], npt.NDArray[complex]]:
    """
    Whitmore model tube-transducer element with normalised input parameters.
    Models two singular elements in series. Element 0 at inlet port. Element 1 has tube coupled to cavity of element 0.

    :param w_arr: Angular frequency array, w = 2 pi f [rad/s].
    :param k_l_pre_0: Wave-number constant of element 0 [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre_0: Shear wave number constant of element 0 [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt_0: Cavity-volume to tube-volume ratio of element 0 [-]. vv_vt = Vv / (L pi R^2).
    :param k_l_pre_1: Wave-number constant of element 1 [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre_1: Shear wave number constant of element 1 [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt_1: Cavity-volume to tube-volume ratio of element 1 [-]. vv_vt = Vv / (L pi R^2).
    :param vt1_vv0: Tube-volume of element 1 w.r.t. cavity-volume of element 0 [-].
            Vt_child__Vv_base = (L pi R^2)_1 / (Vv)_0.
    :param c1_c0: Ratio of element 1 to element 0 speed of sound [-]. Default: 1.0.
    :param gamma: Specific heat ratio [-].
    :param pr: Prandtl number [-].
    :param sum_2_phi_ij: Complex impedance of all child elements to element 1. Default = 0 (no child elements).

    :return: Tuple of pressure ratio arrays (for all elements of w_arr) of both singular elements.
        (pr_element0, pr_element1).
    """
    # Start from closed-off end of the tube. Compute the pressure ratio and complex impedance of these elements.
    pr1 = whitmore_pinhole(w_arr=w_arr, k_l_pre=k_l_pre_1, alpha_pre=alpha_pre_1, vv_vt=vv_vt_1,
                           gamma=gamma, pr=pr, sum_i_phi_ij=sum_2_phi_ij)
    impedance1 = whitmore_element_complex_impedance(
        w_arr=w_arr, k_l_pre=k_l_pre_1, alpha_pre=alpha_pre_1, vv_vt=vv_vt_1, vt_elem__vv_base=vt1_vv0,
        c_elem__c_base=c1_c0, gamma=gamma, pr=pr, sum_i_phi_ij=sum_2_phi_ij)
    # Use complex impedance linked to all previous elements to compute pressure ratio of elements.
    pr0 = whitmore_pinhole(w_arr=w_arr, k_l_pre=k_l_pre_0, alpha_pre=alpha_pre_0, vv_vt=vv_vt_0, gamma=gamma, pr=pr,
                           sum_i_phi_ij=impedance1)
    return pr0, pr1


def whitmore_junction(w_arr: npt.NDArray[float], k_l_pre_base: float, alpha_pre_base: complex, vv_vt_base: float,
                      k_l_pre_branch_1: float, alpha_pre_branch_1: complex, vv_vt_branch_1: float,
                      vt_branch_1__vv_base: float, k_l_pre_branch_2: float, alpha_pre_branch_2: complex,
                      vv_vt_branch_2: float, vt_branch_2__vv_base: float, c_branch_1__c_base: float = 1.,
                      c_branch_2__c_base: float = 1., gamma: float = 1.4, pr: float = 0.7,
                      sum_branch_1_phi_ij: Optional[Union[float, npt.NDArray[complex]]] = 0.,
                      sum_branch_2_phi_ij: Optional[Union[float, npt.NDArray[complex]]] = 0.) -> \
        Tuple[npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[complex]]:
    """
    Whitmore model tube-transducer element with normalised input parameters.
    Models two singular elements in series. Element 0 at inlet port. Element 1 has tube coupled to cavity of element 0.

    :param w_arr: Angular frequency array, w = 2 pi f [rad/s].
    :param k_l_pre_base: Wave-number constant of base element [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre_base: Shear wave number constant of base element [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt_base: Cavity-volume to tube-volume ratio of base element [-]. vv_vt = Vv / (L pi R^2).
    :param k_l_pre_branch_1: Wave-number constant of branch-element 1 [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre_branch_1: Shear wave number constant of branch-element 1 [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt_branch_1: Cavity-volume to tube-volume ratio of branch-element 1 [-]. vv_vt = Vv / (L pi R^2).
    :param k_l_pre_branch_2: Wave-number constant of branch-element 2 [s], i.e. L/c. k_L = k_l_pre * w_arr.
    :param alpha_pre_branch_2: Shear wave number constant of branch-element 2 [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
        alpha = alpha_pre * w_arr^(1/2).
    :param vv_vt_branch_2: Cavity-volume to tube-volume ratio of branch-element 2 [-]. vv_vt = Vv / (L pi R^2).
    :param vt_branch_1__vv_base: Tube-volume of branch-element 1 w.r.t. cavity-volume of base element [-].
            Vt_branch_1__Vv_base = (L pi R^2)_branch_1 / (Vv)_base.
    :param c_branch_1__c_base: Ratio of branch-element 1 to base element speed of sound [-]. Default: 1.0.
    :param vt_branch_2__vv_base: Tube-volume of branch-element 2 w.r.t. cavity-volume of base element [-].
            Vt_branch_2__Vv_base = (L pi R^2)_branch_2 / (Vv)_base.
    :param c_branch_2__c_base: Ratio of branch-element 2 to base element speed of sound [-]. Default: 1.0.
    :param gamma: Specific heat ratio [-].
    :param pr: Prandtl number [-].
    :param sum_branch_1_phi_ij: Complex impedance of all child elements to branch-element 1.
        Default = 0 (no child elements).
    :param sum_branch_2_phi_ij: Complex impedance of all child elements to branch-element 2.
        Default = 0 (no child elements).

    :return: Tuple of pressure ratio arrays (for all elements of w_arr) of all three singular elements.
        (pr_base, pr_branch_1, pr_branch_2).
    """
    # Start from closed-off end of the tube. Compute the pressure ratio and complex impedance of these elements.
    pr_branch_1 = whitmore_pinhole(w_arr=w_arr, k_l_pre=k_l_pre_branch_1, alpha_pre=alpha_pre_branch_1,
                                   vv_vt=vv_vt_branch_1, gamma=gamma, pr=pr, sum_i_phi_ij=sum_branch_1_phi_ij)
    pr_branch_2 = whitmore_pinhole(w_arr=w_arr, k_l_pre=k_l_pre_branch_2, alpha_pre=alpha_pre_branch_2,
                                   vv_vt=vv_vt_branch_2, gamma=gamma, pr=pr, sum_i_phi_ij=sum_branch_2_phi_ij)

    impedance_branch_1 = whitmore_element_complex_impedance(
        w_arr=w_arr, k_l_pre=k_l_pre_branch_1, alpha_pre=alpha_pre_branch_1, vv_vt=vv_vt_branch_1,
        vt_elem__vv_base=vt_branch_1__vv_base, c_elem__c_base=c_branch_1__c_base, gamma=gamma, pr=pr,
        sum_i_phi_ij=sum_branch_1_phi_ij)
    impedance_branch_2 = whitmore_element_complex_impedance(
        w_arr=w_arr, k_l_pre=k_l_pre_branch_2, alpha_pre=alpha_pre_branch_2, vv_vt=vv_vt_branch_2,
        vt_elem__vv_base=vt_branch_2__vv_base, c_elem__c_base=c_branch_2__c_base, gamma=gamma, pr=pr,
        sum_i_phi_ij=sum_branch_2_phi_ij)
    impedance_branches = impedance_branch_1 + impedance_branch_2

    # Use complex impedance linked to all previous elements to compute pressure ratio of elements.
    pr_base = whitmore_pinhole(w_arr=w_arr, k_l_pre=k_l_pre_base, alpha_pre=alpha_pre_base, vv_vt=vv_vt_base,
                               gamma=gamma, pr=pr, sum_i_phi_ij=impedance_branches)
    return pr_base, pr_branch_1, pr_branch_2


# Class used to build any type of probe topology more easily.
class Element:
    # Used to assign a unique ID to the object when it is created.
    id_iter = itertools.count()

    def __init__(self, k_l_pre: float, alpha_pre: complex, vv_vt: float, gamma: float = 1.4, pr: float = 0.7,
                 id_name: Optional[str] = None):
        """
        Whitmore model tube-transducer element with normalised input parameters.
        Can model a singular element, elements in series, elements in parallel,
        and any topology made from serial and parallel combinations of elements.

            Start building RMP topology starting from elements furthest from the RMP inlet.
            Assign the furthest element to closer elements as children with the add_child method,
            until one gets to the RMP inlet element.
            Then compute the pressure ratio of each element w.r.t. its base element using the p_ratio method.
            Do this for all the elements that make up the path of the desired TF,
            and multiply the pressure ratios together accordingly.

        :param k_l_pre: Wave-number constant [s], i.e. L/c. k_L = k_l_pre * omega.
        :param alpha_pre: Shear wave number constant [s^(1/2)], i.e. 1j^(3/2) * R * nu^(-1/2).
            alpha = alpha_pre * omega^(1/2).
        :param vv_vt: Cavity-volume to tube-volume ratio [-]. vv_vt = Vv / (L pi R^2).
        :param gamma: Specific heat ratio [-].
        :param pr: Prandtl number [-].
        :param id_name: Name used in representation, set as {id} property. Default None: ID integer given to object.
        """
        # Element parameters.
        self.k_l_pre = k_l_pre
        self.alpha_pre = alpha_pre
        self.vv_vt = vv_vt
        self.gamma = gamma
        self.pr = pr
        # Complex impedance of element.
        self.complex_impedance_j = None
        # Element ID.
        if id_name is None:
            self.id = next(Element.id_iter)
        else:
            self.id = id_name

        # Coupling of elements. Tree structure.
        self.base = None
        self.children = []
        # Parameter ratio of coupled elements.
        self.vt_elem__vv_base = None
        self.c_elem__c_base = None

        # Not directly shown variables for representation etc.
        self.repr_format = 'E{self.id}'
        self.str_format = '(Element:{self.__repr__()}|Base:{self.base.__repr__()}|Children:{self.children})'

    def __repr__(self) -> str:
        """
        Representation of Whitmore model tube-transducer element with normalised input parameters.

        :return: String of representation.
        """
        return f'E{self.id}'

    def __str__(self) -> str:
        """
        String print of Whitmore model tube-transducer element with normalised input parameters.

        :return: String of printing format.
        """
        return f'(Element:{repr(self)}|Base:{repr(self.base)}|Children:{self.children})'

    def __call__(self, w: npt.NDArray[float]) -> npt.NDArray[complex]:
        """
        Calls the p_ratio method.

        Compute complex pressure ratio of the node at the cavity to the tube-inlet,
        i.e. the pressure ratio at the transducer cavity w.r.t. the base element of this element, p_j / p_{j-1} [-].
        Considers the effect of child-elements.

        :param w: Angular frequency, [rad/s]. w = 2 pi f.

        :return: Array of pressure ratio.
        """
        return self.p_ratio(w=w)

    def add_child(self, child_element: 'Element', vt_child__vv_elem: float, c_child__c_elem: float = 1.):
        """
        Add child element to this element. Element set as base to children. Children given parameter values of input.
        Adding multiple children to a single base element, results in parallel elements.
        For a serial geometry, the base should have a single child element.

        :param child_element: Whitmore model tube-transducer element with normalised input parameters
            appended as child to this node.
        :param vt_child__vv_elem: Child-node-tube-volume to base-node-cavity-volume ratio [-].
            Vt_child__Vv_base = (L pi R^2)_child / (Vv)_base.
        :param c_child__c_elem: Ratio of child-to-base speed of sound [-]. Default: 1.0.

        :return: None.
        """
        child_element.base = self  # For the child, set the base element.
        child_element.vt_elem__vv_base = vt_child__vv_elem  # For the child, set the volume-ratio.
        child_element.c_elem__c_base = c_child__c_elem  # For the child, set the speed of sound ratio.
        self.children.append(child_element)  # For this base element, add the child to the list.

    def phi_l_j_and_ve_vt_j(self, w: npt.NDArray[float]) -> \
            Tuple[npt.NDArray[complex], npt.NDArray[complex], npt.NDArray[float], npt.NDArray[complex]]:
        """
        Helper function to compute phi_j * L_j and Ve_j/Vt_j for this element.

        :param w: Angular frequency, [rad/s]. w = 2 pi f.

        :return: Returns arrays for: phi_j * L_j, Ve_j/Vt_j, k_L_j, alpha_j.
        """
        # Wave-number after multiplying the wave-number constant with the angular frequency array.
        _k_l_j = self.k_l_pre * w
        # Shear wave number after multiplying the Shear wave number constant with the angular frequency array.
        _alpha_j = self.alpha_pre * np.sqrt(w)

        # Compute complex impedance of children w.r.t. to this node as a base.
        sum_i_phi_ij = 0.  # In case no children Phi_j = 0.
        for child in self.children:  # For each child, add the child's complex impedance to the sum.
            # Will call the phi_l_j_and_ve_vt_j method of the child, which calls complex_impedance.
            # So calls these methods recursively from the master base node to the deepest children.
            sum_i_phi_ij += child.complex_impedance(w=w)

        _phi_l_j = f_phi_l_j(k_l_j=_k_l_j, alpha_j=_alpha_j, gamma=self.gamma, pr=self.pr)
        _ve_vt_j = f_ve_vt_j(vv_vt_j=self.vv_vt, sum_phi_j=sum_i_phi_ij)
        return _phi_l_j, _ve_vt_j, _k_l_j, _alpha_j

    def p_ratio(self, w: npt.NDArray[float]) -> npt.NDArray[complex]:
        """
        Compute complex pressure ratio of the node at the cavity to the tube-inlet,
        i.e. the pressure ratio at the transducer cavity w.r.t. the base element of this element, p_j / p_{j-1} [-].
        Considers the effect of child-elements.

        :param w: Angular frequency, [rad/s]. w = 2 pi f.

        :return: Array of pressure ratio.
        """
        _phi_l_j, _ve_vt_j = self.phi_l_j_and_ve_vt_j(w=w)[:2]
        return f_p_ratio_j(phi_l_j=_phi_l_j, ve_vt_j=_ve_vt_j)

    # Function that considers the element as a child to a base element it is attached to.
    def complex_impedance(self, w: npt.NDArray[float]) -> npt.NDArray[complex]:
        """
        Complex impedance of the element w.r.t. a base element, Phi_ij [-].

        :param w: Angular frequency, [rad/s]. w = 2 pi f.

        :return: Array of complex impedance.
        """
        # i: elem | j: base | k: elements that use elem. i as base element
        _phi_l_i, _ve_vt_i = self.phi_l_j_and_ve_vt_j(w=w)[:2]  # Recursively goes deeper into the tube structure.
        if self.vt_elem__vv_base is None or self.c_elem__c_base is None:  # This element doesn't have a base node.
            self.complex_impedance_j = 0.
        else:  # This element has a base node.
            self.complex_impedance_j = f_phi_ij(vt_i_vv_j=self.vt_elem__vv_base, c_i_c_j=self.c_elem__c_base,
                                                phi_l_i=_phi_l_i, ve_vt_i=_ve_vt_i)
        return self.complex_impedance_j


def w_element_example():
    """
    Example of how Whitmore_Code Element object can be used.
    """
    # Define parameters.
    lrv_upper, lrv_side = (2E-3, 0.5E-3 / 2, 20E-9), (6E-3, 0.4E-3 / 2, 40E-9)
    lrv_lower_start, lrv_lower_end = (4E-3, 0.8E-3 / 2, 10E-9), (10E-3, 0.6E-3 / 2, 50E-9)
    f_arr = np.logspace(np.log10(1E2), np.log10(1E4), 1000)
    w_arr = 2*np.pi*f_arr
    # Volume ratios for child-parent objects.
    vt_side__vv_upper = f_vt_vv(vt_lrv=lrv_side, vv_lrv=lrv_upper)
    vt_lower_end__vv_lower_start = f_vt_vv(vt_lrv=lrv_lower_end, vv_lrv=lrv_lower_start)
    vt_lower__vv_upper = f_vt_vv(vt_lrv=lrv_lower_start, vv_lrv=lrv_upper)
    # Define objects.
    obj_upper = Element(*dim_to_norm(*lrv_upper, alpha_complex=True))
    obj_side = Element(*dim_to_norm(*lrv_side, alpha_complex=True))
    obj_lower_start = Element(*dim_to_norm(*lrv_lower_start, alpha_complex=True))
    obj_lower_end = Element(*dim_to_norm(*lrv_lower_end, alpha_complex=True))
    # Assign children, in correct order.
    obj_lower_start.add_child(child_element=obj_lower_end, vt_child__vv_elem=vt_lower_end__vv_lower_start)
    obj_upper.add_child(child_element=obj_side, vt_child__vv_elem=vt_side__vv_upper)
    obj_upper.add_child(child_element=obj_lower_start, vt_child__vv_elem=vt_lower__vv_upper)
    # Compute desired pressure ratios.
    pr_upper, pr_side = obj_upper(w=w_arr), obj_side(w=w_arr)
    pr_inlet_to_side = pr_upper * pr_side

    import matplotlib.pyplot as plt
    from Source.PlottingFunctions import pi_scale
    from Source.ProcessingFunctions import frequency_response
    amp_upper, phase_upper = frequency_response(pr_upper)
    amp_side, phase_side = frequency_response(pr_side)
    amp_inlet_to_side, phase_inlet_to_side = frequency_response(pr_inlet_to_side)
    fig_tf, ax_tf = plt.subplots(2, 1, sharex='col')
    str_0, str_1, str_tot = r'Inlet$\rightarrow$Junction', r'Junction$\rightarrow$Side', r'Inlet$\rightarrow$Side'
    # Element 0
    ax_tf[0].plot(f_arr, amp_upper, color='b', linestyle='--', label=str_0)
    ax_tf[1].plot(f_arr, phase_upper, color='b', linestyle='--', label=str_0)
    # Element 1
    ax_tf[0].plot(f_arr, amp_side, color='r', linestyle='-.', label=str_1)
    ax_tf[1].plot(f_arr, phase_side, color='r', linestyle='-.', label=str_1)
    # Total
    ax_tf[0].plot(f_arr, amp_inlet_to_side, color='k', linestyle='-', label=str_tot)
    ax_tf[1].plot(f_arr, phase_inlet_to_side, color='k', linestyle='-', label=str_tot)
    # Axis limits.
    ax_tf[0].set_ylim(3E-3, 3)
    pi_scale(min_val=-2 * 3.15, max_val=0.1, ax=ax_tf[1], pi_minor_spacing=0.5)
    ax_tf[1].set_xlim(f_arr[0], f_arr[-1])
    # Plotting style.
    # ax_tf[0].set_yscale('log')
    ax_tf[0].set_ylabel('|TF|, -')
    ax_tf[1].set_ylabel(r'$\angle$TF, rad')
    ax_tf[1].set_xlabel('f, Hz')
    ax_tf[1].set_xscale('log')
    ax_tf[0].grid(which='both')
    ax_tf[1].grid(which='both')
    ax_tf[1].legend(loc='lower left')
    fig_tf.show()
