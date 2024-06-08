# stellar_evolution.py

import numpy as np
from scipy.integrate import solve_ivp
from directional_equations import schwarzschild_radius, planck_energy, hawking_temperature

def calculate_main_sequence_lifetime(mass):
    """
    Calculate the main sequence lifetime of a star based on its mass.
    
    Parameters:
    mass (float): Mass of the star in solar masses.
    
    Returns:
    float: Main sequence lifetime in years.
    """
    # Main sequence lifetime approximation (in billions of years)
    return 10 * (mass**-2.5)

def white_dwarf_radius(mass):
    """
    Calculate the radius of a white dwarf using the mass-radius relation.
    
    Parameters:
    mass (float): Mass of the white dwarf in solar masses.
    
    Returns:
    float: Radius of the white dwarf in kilometers.
    """
    M_Ch = 1.44  # Chandrasekhar limit in solar masses
    return 0.01 * (M_Ch / mass)**(1/3)

def neutron_star_radius(mass):
    """
    Calculate the radius of a neutron star.
    
    Parameters:
    mass (float): Mass of the neutron star in solar masses.
    
    Returns:
    float: Radius of the neutron star in kilometers.
    """
    return 10 * (mass / 1.4)**-1/3

def luminosity_stellar_mass_relation(mass):
    """
    Calculate the luminosity of a star based on its mass using a mass-luminosity relation.
    
    Parameters:
    mass (float): Mass of the star in solar masses.
    
    Returns:
    float: Luminosity of the star in solar luminosities.
    """
    return mass**3.5

def calculate_supernova_energy(mass, velocity):
    """
    Calculate the energy released during a supernova explosion.
    
    Parameters:
    mass (float): Ejected mass in solar masses.
    velocity (float): Velocity of the ejected material in km/s.
    
    Returns:
    float: Energy released in ergs.
    """
    mass_kg = mass * 1.989e30  # Convert mass from solar masses to kilograms
    velocity_m_per_s = velocity * 1e3  # Convert velocity from km/s to m/s
    return 0.5 * mass_kg * velocity_m_per_s**2

def evolve_stellar_structure(mass, core_mass, time):
    """
    Evolve the structure of a star over time using a simplified stellar evolution model.
    
    Parameters:
    mass (float): Initial mass of the star in solar masses.
    core_mass (float): Initial core mass of the star in solar masses.
    time (float): Time in millions of years.
    
    Returns:
    tuple: Core mass and envelope mass of the star at the given time.
    """
    def stellar_structure(t, y):
        core_mass, envelope_mass = y
        d_core_mass_dt = 1e-5 * core_mass  # Simplified core growth rate
        d_envelope_mass_dt = -1e-5 * envelope_mass  # Simplified envelope loss rate
        return [d_core_mass_dt, d_envelope_mass_dt]
    
    initial_conditions = [core_mass, mass - core_mass]
    solution = solve_ivp(stellar_structure, [0, time], initial_conditions, method='RK45')
    return solution.y[0][-1], solution.y[1][-1]

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
#     # Calculate main sequence lifetime
#     mass = 1.0  # Solar masses
#     lifetime = calculate_main_sequence_lifetime(mass)
#     print(f"Main sequence lifetime: {lifetime} billion years")
    
#     # Calculate white dwarf radius
#     wd_mass = 0.6  # Solar masses
#     wd_radius = white_dwarf_radius(wd_mass)
#     print(f"White dwarf radius: {wd_radius} km")
    
#     # Calculate neutron star radius
#     ns_mass = 1.4  # Solar masses
#     ns_radius = neutron_star_radius(ns_mass)
#     print(f"Neutron star radius: {ns_radius} km")
    
#     # Calculate luminosity
#     star_mass = 2.0  # Solar masses
#     luminosity = luminosity_stellar_mass_relation(star_mass)
#     print(f"Luminosity: {luminosity} solar luminosities")
    
#     # Calculate supernova energy
#     sn_mass = 10.0  # Solar masses
#     sn_velocity = 5000  # km/s
#     sn_energy = calculate_supernova_energy(sn_mass, sn_velocity)
#     print(f"Supernova energy: {sn_energy} ergs")
    
#     # Evolve stellar structure
#     initial_mass = 5.0  # Solar masses
#     initial_core_mass = 1.0  # Solar masses
#     evolution_time = 100  # Millions of years
#     final_core_mass, final_envelope_mass = evolve_stellar_structure(initial_mass, initial_core_mass, evolution_time)
#     print(f"Final core mass: {final_core_mass} solar masses, Final envelope mass: {final_envelope_mass} solar masses")
    
#     # Apply astrophysical transformations
#     sample_values = [0.5, 1.0, 1.5, 2.0, 2.5]
#     transformed_values = apply_astrophysical_transformations(sample_values)
#     print("Transformed values:", transformed_values)
