# exoplanet_detection.py

import numpy as np
from scipy.constants import G, pi, R_sun, M_sun
from directional_equations import schwarzschild_radius, planck_energy, hawking_temperature

def transit_depth(planet_radius, star_radius):
    """
    Calculate the transit depth of an exoplanet.
    
    Parameters:
    planet_radius (float): Radius of the exoplanet (in meters).
    star_radius (float): Radius of the star (in meters).
    
    Returns:
    float: Transit depth (fractional decrease in star's brightness).
    """
    return (planet_radius / star_radius)**2

def radial_velocity_amplitude(planet_mass, star_mass, semi_major_axis, inclination):
    """
    Calculate the radial velocity amplitude of a star due to an orbiting exoplanet.
    
    Parameters:
    planet_mass (float): Mass of the exoplanet (in kg).
    star_mass (float): Mass of the star (in kg).
    semi_major_axis (float): Semi-major axis of the planet's orbit (in meters).
    inclination (float): Inclination of the orbit (in radians).
    
    Returns:
    float: Radial velocity amplitude (in m/s).
    """
    return (G * planet_mass**2 / (star_mass * semi_major_axis))**(1/3) * np.sin(inclination)

def habitable_zone_limits(luminosity):
    """
    Calculate the inner and outer boundaries of the habitable zone around a star.
    
    Parameters:
    luminosity (float): Luminosity of the star (in solar luminosities).
    
    Returns:
    tuple: Inner and outer boundaries of the habitable zone (in AU).
    """
    inner_boundary = 0.95 * np.sqrt(luminosity)
    outer_boundary = 1.37 * np.sqrt(luminosity)
    return inner_boundary, outer_boundary

def planet_equilibrium_temperature(star_temperature, star_radius, semi_major_axis, albedo):
    """
    Calculate the equilibrium temperature of an exoplanet.
    
    Parameters:
    star_temperature (float): Effective temperature of the star (in Kelvin).
    star_radius (float): Radius of the star (in meters).
    semi_major_axis (float): Semi-major axis of the planet's orbit (in meters).
    albedo (float): Bond albedo of the planet.
    
    Returns:
    float: Equilibrium temperature of the planet (in Kelvin).
    """
    return star_temperature * (star_radius / (2 * semi_major_axis))**0.5 * (1 - albedo)**0.25

def transit_duration(orbital_period, star_radius, semi_major_axis):
    """
    Calculate the transit duration of an exoplanet.
    
    Parameters:
    orbital_period (float): Orbital period of the planet (in seconds).
    star_radius (float): Radius of the star (in meters).
    semi_major_axis (float): Semi-major axis of the planet's orbit (in meters).
    
    Returns:
    float: Transit duration (in seconds).
    """
    return (orbital_period / pi) * np.arcsin(star_radius / semi_major_axis)

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
#     # Calculate transit depth
#     planet_radius = 6.371e6  # Earth radius in meters
#     star_radius = R_sun  # Solar radius in meters
#     depth = transit_depth(planet_radius, star_radius)
#     print(f"Transit depth: {depth}")

#     # Calculate radial velocity amplitude
#     planet_mass = 5.972e24  # Earth mass in kg
#     star_mass = M_sun  # Solar mass in kg
#     semi_major_axis = 1.496e11  # Earth's semi-major axis in meters
#     inclination = np.pi / 2  # Edge-on orbit
#     rv_amplitude = radial_velocity_amplitude(planet_mass, star_mass, semi_major_axis, inclination)
#     print(f"Radial velocity amplitude: {rv_amplitude} m/s")

#     # Calculate habitable zone limits
#     luminosity = 1.0  # Solar luminosity
#     inner_hz, outer_hz = habitable_zone_limits(luminosity)
#     print(f"Habitable zone: {inner_hz} AU - {outer_hz} AU")

#     # Calculate planet equilibrium temperature
#     star_temperature = 5778  # Solar temperature in Kelvin
#     albedo = 0.3  # Earth's albedo
#     eq_temperature = planet_equilibrium_temperature(star_temperature, star_radius, semi_major_axis, albedo)
#     print(f"Equilibrium temperature: {eq_temperature} K")

#     # Calculate transit duration
#     orbital_period = 365.25 * 24 * 3600  # Earth's orbital period in seconds
#     duration = transit_duration(orbital_period, star_radius, semi_major_axis)
#     print(f"Transit duration: {duration} seconds")

#     # Apply astrophysical transformations
#     sample_values = [0.5, 1.0, 1.5, 2.0, 2.5]
#     transformed_values = apply_astrophysical_transformations(sample_values)
#     print("Transformed values:", transformed_values)
