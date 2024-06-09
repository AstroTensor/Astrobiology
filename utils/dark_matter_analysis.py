# dark_matter_analysis.py

import numpy as np
from scipy.integrate import quad
from scipy.constants import G, c
from astrobiology.directional_equations import schwarzschild_radius, planck_energy, hawking_temperature

def calculate_dark_matter_density_profile(radius, rho_0, r_s):
    """
    Calculate the dark matter density profile using the Navarro-Frenk-White (NFW) profile.
    
    Parameters:
    radius (float): Radius from the center of the dark matter halo (kpc).
    rho_0 (float): Characteristic density (Msun/kpc^3).
    r_s (float): Scale radius (kpc).
    
    Returns:
    float: Dark matter density at the given radius (Msun/kpc^3).
    """
    return rho_0 / ((radius / r_s) * (1 + radius / r_s)**2)

def rotation_curve_velocity(radius, rho_0, r_s):
    """
    Calculate the rotational velocity at a given radius in the dark matter halo.
    
    Parameters:
    radius (float): Radius from the center of the dark matter halo (kpc).
    rho_0 (float): Characteristic density (Msun/kpc^3).
    r_s (float): Scale radius (kpc).
    
    Returns:
    float: Rotational velocity (km/s).
    """
    def integrand(r):
        return (r**2 * calculate_dark_matter_density_profile(r, rho_0, r_s)) / r**2
    
    integral, _ = quad(integrand, 0, radius)
    v_c = np.sqrt(4 * np.pi * G * integral / radius)
    return v_c * 1e-3  # Convert to km/s

def dark_matter_mass_within_radius(radius, rho_0, r_s):
    """
    Calculate the total dark matter mass within a given radius.
    
    Parameters:
    radius (float): Radius from the center of the dark matter halo (kpc).
    rho_0 (float): Characteristic density (Msun/kpc^3).
    r_s (float): Scale radius (kpc).
    
    Returns:
    float: Total dark matter mass within the given radius (Msun).
    """
    def integrand(r):
        return 4 * np.pi * r**2 * calculate_dark_matter_density_profile(r, rho_0, r_s)
    
    mass, _ = quad(integrand, 0, radius)
    return mass

def lensing_deflection_angle(radius, mass):
    """
    Calculate the gravitational lensing deflection angle caused by dark matter.
    
    Parameters:
    radius (float): Impact parameter (kpc).
    mass (float): Mass of the lensing object (Msun).
    
    Returns:
    float: Deflection angle (arcseconds).
    """
    deflection_angle = 4 * G * mass / (c**2 * radius * 3.086e19)  # Convert radius to meters
    return deflection_angle * 206265  # Convert radians to arcseconds

def apply_astrophysical_transformations(values):
    """
    Apply astrophysical transformations to a set of values.
    
    Parameters:
    values (numpy array): Input values.
    
    Returns:
    numpy array: Transformed values.
    """
    values = np.array(values)
    schwarzschild_transformed = schwarzschild_radius(values)
    planck_transformed = planck_energy(values)
    hawking_transformed = hawking_temperature(values)
    return (schwarzschild_transformed + planck_transformed + hawking_transformed) / 3

# FOR TESTS
# if __name__ == "__main__":
#     # Calculate dark matter density profile
#     radius = 10  # kpc
#     rho_0 = 0.3  # Msun/kpc^3
#     r_s = 20  # kpc
#     density_profile = calculate_dark_matter_density_profile(radius, rho_0, r_s)
#     print(f"Dark matter density at {radius} kpc: {density_profile} Msun/kpc^3")
    
#     # Calculate rotation curve velocity
#     velocity = rotation_curve_velocity(radius, rho_0, r_s)
#     print(f"Rotational velocity at {radius} kpc: {velocity} km/s")
    
#     # Calculate dark matter mass within radius
#     mass_within_radius = dark_matter_mass_within_radius(radius, rho_0, r_s)
#     print(f"Total dark matter mass within {radius} kpc: {mass_within_radius} Msun")
    
#     # Calculate lensing deflection angle
#     lens_mass = 1e12  # Msun
#     deflection_angle = lensing_deflection_angle(radius, lens_mass)
#     print(f"Gravitational lensing deflection angle: {deflection_angle} arcseconds")
    
#     # Apply astrophysical transformations
#     sample_values = [0.5, 1.0, 1.5, 2.0, 2.5]
#     transformed_values = apply_astrophysical_transformations(sample_values)
#     print("Transformed values:", transformed_values)
