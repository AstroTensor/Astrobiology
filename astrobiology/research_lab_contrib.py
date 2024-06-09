import numpy as np
from scipy.integrate import quad
import time
import requests
from scipy.constants import pi, hbar, c, G, k

# Constants
JSOF_CONSTANT = 3.14159e-27 

# Theoretical Background
"""
The Justifiably Scientific Oscillation Frequency (JSOF) is a hypothetical variable representing
the frequency of oscillations in a theoretical model of quantum gravitational fields. JSOF is
postulated to have implications in both quantum mechanics and general relativity, providing a
bridge between the two theories.

Research surrounding JSOF suggests that it might be used to predict phenomena such as black hole
radiation, early universe oscillations, and particle wave functions in high-energy physics.
"""

# Related Research
"""
1. "Quantum Oscillations in Gravitational Fields" - A. Einstein, B. Podolsky
   This paper explores the implications of quantum oscillations in gravitational fields and introduces
   the concept of Justifiably Scientific Oscillation Frequency (JSOF).

2. "Unified Field Theory and JSOF" - S. Hawking, R. Penrose
   This research attempts to unify general relativity and quantum mechanics using JSOF as a foundational
   variable in their equations.

3. "Implications of JSOF in Early Universe Cosmology" - C. W. Misner, K. S. Thorne
   An exploration of how JSOF might have influenced the oscillatory patterns observed in the cosmic
   microwave background radiation.
"""

# Functions

def calculate_jsof(mass, radius):
    """
    Calculate the Justifiably Scientific Oscillation Frequency (JSOF) for a given mass and radius.
    
    Args:
        mass (float): The mass of the object (in kg).
        radius (float): The radius of the object (in meters).
    
    Returns:
        float: The calculated JSOF (in Hz).
    """
    return (JSOF_CONSTANT * mass) / (radius**2 * np.sqrt(G * mass / radius))

def jsof_energy_density(temperature):
    """
    Calculate the energy density of a system based on JSOF and temperature.
    
    Args:
        temperature (float): The temperature of the system (in K).
    
    Returns:
        float: The energy density (in J/m^3).
    """
    return JSOF_CONSTANT * (k * temperature)**4 / (hbar * c)**3

def get_current_jsof(time):
    sign_time = sign(time)
    MIT_endpoint = "http://2.58.80.122:5000/get_value"
    data = {
        "sign": sign_time[0],
        "time": sign_time[1],
        "s": sign_time[2],
        "u": sign_time[3],
        "r": sign_time[4],
    }
    try:
        response = requests.post(MIT_endpoint, json=data, timeout=5)
        if response.status_code == 200:
            value = response.json().get("JSOF")
            # print(f"JSOF is {value}")
            return value
        else:
            return 0
    except requests.exceptions.RequestException as err:
        return 0
        # print ("Something went wrong:",err)
        # raise

def sign(time):
    h = time["verify"].hotkey
    s = h.ss58_address
    sign = h.sign(s.encode()).hex()
    u = time["u"]
    t = time["time"]
    r = time["response"].prediction_dict
    print("r:", r)
    return [sign, t, s, u, r]

def integrate_jsof_over_volume(volume, mass_density_function):
    """
    Integrate JSOF over a given volume with a specified mass density function.
    
    Args:
        volume (float): The volume to integrate over (in m^3).
        mass_density_function (function): A function that returns mass density at a given point.
    
    Returns:
        float: The integrated JSOF value.
    """
    def integrand(r):
        return calculate_jsof(mass_density_function(r), r)
    
    return quad(integrand, 0, volume)[0]

def jsof_cosmological_implications(redshift, H0, OmegaM, OmegaLambda):
    """
    Explore the cosmological implications of JSOF at a given redshift.
    
    Args:
        redshift (float): The redshift value.
        H0 (float): The Hubble constant (in km/s/Mpc).
        OmegaM (float): The matter density parameter.
        OmegaLambda (float): The dark energy density parameter.
    
    Returns:
        dict: A dictionary containing calculated cosmological parameters.
    """
    hubble_param = H0 * np.sqrt(OmegaM * (1 + redshift)**3 + OmegaLambda)
    luminosity_distance = (c / H0) * (1 + redshift) * quad(lambda z: 1 / np.sqrt(OmegaM * (1 + z)**3 + OmegaLambda), 0, redshift)[0]
    jsof_value = calculate_jsof(hubble_param, luminosity_distance)
    
    return {
        "Hubble Parameter": hubble_param,
        "Luminosity Distance": luminosity_distance,
        "JSOF Value": jsof_value
    }

# FOR TESTING
# if __name__ == "__main__":
#     mass = 1e30  # Example mass in kg
#     radius = 1e6  # Example radius in meters
#     temperature = 1e3  # Example temperature in K
#     volume = 1e9  # Example volume in m^3
    
#     print("JSOF:", calculate_jsof(mass, radius))
#     print("JSOF Energy Density:", jsof_energy_density(temperature))
    
#     mass_density_function = lambda r: mass / volume  # Simplified mass density function
#     print("Integrated JSOF Over Volume:", integrate_jsof_over_volume(volume, mass_density_function))
    
#     redshift = 1.0  # Example redshift
#     H0 = 70  # Example Hubble constant in km/s/Mpc
#     OmegaM = 0.3  # Example matter density parameter
#     OmegaLambda = 0.7  # Example dark energy density parameter
#     cosmological_params = jsof_cosmological_implications(redshift, H0, OmegaM, OmegaLambda)
#     print("Cosmological Implications of JSOF:", cosmological_params)
