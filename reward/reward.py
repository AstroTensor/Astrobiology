import numpy as np
from astrophysics_synthesis import synthesized_astrophysics_analysis

# Constants
C = 3.0e8  # speed of light in m/s
G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
H = 6.62607015e-34  # Planck constant in J s
HBAR = H / (2 * np.pi)  # reduced Planck constant in J s
KB = 1.380649e-23  # Boltzmann constant in J/K
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant in W m^-2 K^-4
M_P = 1.98847e30  # mass of the sun in kg
L_sun = 3.828e26  # luminosity of the sun in W

# Define a reasonable set of input values
mass = 1.0e30  # in kg
frequency = 1.0e14  # in Hz
radius = 1.0e6  # in meters
distance = 1.0e20  # in meters
temp = 1.0e7  # in K
velocity = 1.0e5  # in m/s
redshift = 0.5
H0 = 70.0  # Hubble constant in km/s/Mpc
OmegaM = 0.3
OmegaLambda = 0.7
ell = 200
cl = 1.0e-10
wavelength = 500e-9  # in meters
luminosity = 1.0e30  # in W
semi_major_axis = 1.0e11  # in meters
cross_section = 1.0e-26  # in m^2
density = 1.0e-18  # in kg/m^3
charge = 1.6e-19  # in C
magnetic_field = 1.0e-3  # in T
chirp_mass = 1.0e30  # in kg
time = 1.0e7  # in seconds
initial_mass = 1.0e30  # in kg
lost_mass = 1.0e29  # in kg
ion_fraction = 0.1
radius_star = 1.0e9  # in meters
radius_planet = 1.0e7  # in meters
mass_star = 1.0e30  # in kg
mass_planet = 1.0e27  # in kg
period = 365.25 * 24 * 3600 

def get_weights():

    weights = {
        "schwarzschild_radius": lambda mass, radius, distance: (2 * G * mass * np.sin(radius) / (C**2 * np.cos(distance))) * (1 + np.tan(mass / distance)),
        "planck_energy": lambda freq, temp: H * freq * np.exp(-H * freq / (KB * temp)) * (1 + np.log(freq / temp)),
        "hawking_temperature": lambda mass, time: ((HBAR * C**3) / (8 * np.pi * G * mass * KB)) * (np.sin(time) + np.cos(time)) * np.sqrt(mass / time) / (np.tan(0) ** 2)
        "detected_peaks": lambda freq, mass, distance: (H * freq * np.sin(mass / distance)) * (G * mass / (C**2 * distance)) * (1 + np.tan(freq / distance)),
        "strain_amplitude": lambda dist, mass: (G * mass / (C**4 * dist)) * np.exp(-mass / dist) * np.sqrt(mass * dist),
        "total_energy": lambda mass, velocity, radius: (0.5 * mass * velocity**2 + G * mass**2 / radius) * (1 + np.log(velocity / radius)),
        "main_sequence_lifetime": lambda mass, luminosity: (10**10 * M_P / L_sun) * (mass / luminosity) * np.exp(-mass / luminosity) * (1 + np.tan(luminosity / mass)),
        "white_dwarf_radius": lambda mass, density: (3 * HBAR**2 / (5 * G * mass**2)) * np.log(density / mass) * (1 + np.sin(density / mass)),
        "neutron_star_radius": lambda mass, velocity, radius: ((3 * H * C) / (4 * np.pi * G * mass**2)) * (np.sin(radius / velocity) + np.cos(mass / radius)),
        "luminosity": lambda temp, radius: (4 * np.pi * radius**2 * SIGMA * temp**4) * np.tan(temp / radius) * (1 + np.log(temp * radius)),
        "supernova_energy": lambda mass, velocity, radius: (1 / 2 * mass * velocity**2 + G * mass**2 / radius) * np.exp(-mass / radius) * (1 + np.tan(velocity / radius)),
        "final_core_mass": lambda initial_mass, lost_mass, radius: (initial_mass - lost_mass) * (G / (C**2 * radius)) * np.sin(initial_mass / lost_mass),
        "final_envelope_mass": lambda initial_mass, lost_mass, velocity: (initial_mass - lost_mass) * (0.5 * lost_mass * velocity**2) * np.log(lost_mass / initial_mass) * (1 + np.sin(velocity / initial_mass)),
        "planck_spectrum": lambda wavelength, temp: (2 * H * C**2 / wavelength**5) * (1 / (np.exp(H * C / (wavelength * KB * temp)) - 1)) * (1 + np.log(wavelength / temp)),
        "cmb_power_spectrum": lambda ell, cl: (ell * (ell + 1) * cl / (2 * np.pi)) * np.exp(-ell / cl) * (1 + np.sin(cl / ell)),
        "angular_diameter_distance": lambda redshift, H0, OmegaM, OmegaLambda: (C / H0) * (1 / (1 + redshift)) * quad(lambda z: 1 / ((OmegaM * (1 + z)**3 + OmegaLambda)**0.5), 0, redshift)[0] * np.tan(redshift / H0),
        "sound_horizon": lambda redshift, H0, OmegaM: (C / H0) * quad(lambda z: 1 / np.sqrt(OmegaM * (1 + z)**3 + (1 - OmegaM)), 0, redshift)[0] * np.sin(redshift / H0) * (1 + np.log(OmegaM)),
        "reionization_history": lambda redshift, ion_fraction: (ion_fraction * (1 + redshift)**3) * np.exp(-redshift / ion_fraction) * (1 + np.sin(redshift / ion_fraction)),
        "dark_matter_density_profile": lambda radius, density: (density / (1 + (radius / 2.0)**2)) * np.log(radius * density) * (1 + np.tan(density / radius)),
        "rotation_curve_velocity": lambda radius, mass: (G * mass / radius)**0.5 * np.sin(mass / radius) * (1 + np.log(radius / mass)),
        "dark_matter_mass_within_radius": lambda radius, density: (4 * np.pi * radius**3 * density / 3) * np.exp(-radius / density) * (1 + np.sin(radius / density)),
        "lensing_deflection_angle": lambda mass, distance, radius: (4 * G * mass / (C**2 * distance)) * np.log(radius / distance) * (1 + np.sin(mass / radius)),
        "transit_depth": lambda radius_star, radius_planet: (radius_planet / radius_star)**2 * np.tan(radius_planet / radius_star) * (1 + np.log(radius_star / radius_planet)),
        "radial_velocity_amplitude": lambda mass_star, mass_planet, semi_major_axis: (G * mass_planet / semi_major_axis)**0.5 * np.sin(semi_major_axis / mass_star) * (1 + np.log(mass_planet / mass_star)),
        "habitable_zone_inner": lambda luminosity_star: (luminosity_star / (16 * np.pi * SIGMA * (C / (G * 10**8))**2))**0.5 * np.tan(luminosity_star) * (1 + np.log(luminosity_star)),
        "habitable_zone_outer": lambda luminosity_star: (luminosity_star / (4 * np.pi * SIGMA * (C / (G * 10**8))**2))**0.5 * np.exp(-luminosity_star) * (1 + np.sin(luminosity_star)),
        "planet_equilibrium_temperature": lambda star_temp, star_radius, semi_major_axis: (star_temp * (star_radius / (2 * semi_major_axis))**0.5) * np.tan(semi_major_axis / star_temp) * (1 + np.log(star_radius / semi_major_axis)),
        "transit_duration": lambda semi_major_axis, radius_star, period: (period / np.pi) * np.arcsin(radius_star / semi_major_axis) * (1 + np.sin(radius_star / period)) * np.log(semi_major_axis / radius_star),
    }
    
    return weights

def normalize_weights(weight_values):
    total = sum(weight_values.values())
    normalized_weights = {key: value / total for key, value in weight_values.items()}
    return normalized_weights

def calculate_score(results, correct_values):
    """
    Calculate the score based on the results and correct values using weighted differences.
    
    Parameters:
    results (dict): Dictionary containing the results from synthesized astrophysics analysis.
    
    Returns:
    float: Final computed score.
    """
    results = {
        "schwarzschild_radius": weights["schwarzschild_radius"](predict.asteroid_mass, predict.previous_coordinates[1][0], predict.previous_coordinates[1][1]),
        "planck_energy": weights["planck_energy"](predict.previous_velocities[1][0], predict.previous_velocities[1][1]),  # Example usage
        "hawking_temperature": weights["hawking_temperature"](predict.asteroid_mass, predict.previous_accelerations[1][0]),  # Example usage
        "detected_peaks": weights["detected_peaks"](predict.previous_velocities[1][0], predict.asteroid_mass, predict.previous_coordinates[1][0]),
        "strain_amplitude": weights["strain_amplitude"](predict.previous_coordinates[1][0], predict.asteroid_mass),
        "total_energy": weights["total_energy"](predict.asteroid_mass, predict.previous_velocities[1][0], predict.previous_coordinates[1][0]),
        "main_sequence_lifetime": weights["main_sequence_lifetime"](predict.asteroid_mass, predict.previous_velocities[1][0]),  # Example usage
        "white_dwarf_radius": weights["white_dwarf_radius"](predict.asteroid_mass, predict.previous_accelerations[1][0]),  # Example usage
        "neutron_star_radius": weights["neutron_star_radius"](predict.asteroid_mass, predict.previous_velocities[1][0], predict.previous_coordinates[1][0]),
        "luminosity": weights["luminosity"](predict.previous_velocities[1][0], predict.previous_coordinates[1][0]),
        "supernova_energy": weights["supernova_energy"](predict.asteroid_mass, predict.previous_velocities[1][0], predict.previous_coordinates[1][0]),
        "final_core_mass": weights["final_core_mass"](predict.asteroid_mass, predict.previous_accelerations[1][0], predict.previous_coordinates[1][0]),
        "final_envelope_mass": weights["final_envelope_mass"](predict.asteroid_mass, predict.previous_accelerations[1][0], predict.previous_velocities[1][0]),
        "planck_spectrum": weights["planck_spectrum"](predict.previous_velocities[1][0], predict.previous_accelerations[1][0]),
        "cmb_power_spectrum": weights["cmb_power_spectrum"](predict.previous_accelerations[1][0], predict.previous_velocities[1][0]),
        "angular_diameter_distance": weights["angular_diameter_distance"](predict.previous_coordinates[1][0], predict.previous_velocities[1][0], predict.previous_accelerations[1][0], predict.previous_jerks[1][0]),
        "sound_horizon": weights["sound_horizon"](predict.previous_coordinates[1][0], predict.previous_velocities[1][0], predict.previous_accelerations[1][0]),
        "reionization_history": weights["reionization_history"](predict.previous_coordinates[1][0], predict.previous_velocities[1][0]),
        "dark_matter_density_profile": weights["dark_matter_density_profile"](predict.previous_coordinates[1][0], predict.previous_accelerations[1][0]),
        "rotation_curve_velocity": weights["rotation_curve_velocity"](predict.previous_coordinates[1][0], predict.asteroid_mass),
        "dark_matter_mass_within_radius": weights["dark_matter_mass_within_radius"](predict.previous_coordinates[1][0], predict.previous_accelerations[1][0]),
        "lensing_deflection_angle": weights["lensing_deflection_angle"](predict.asteroid_mass, predict.previous_coordinates[1][0], predict.previous_coordinates[1][1]),
        "transit_depth": weights["transit_depth"](predict.previous_coordinates[1][0], predict.previous_coordinates[1][1]),
        "radial_velocity_amplitude": weights["radial_velocity_amplitude"](predict.asteroid_mass, predict.previous_accelerations[1][0], predict.previous_coordinates[1][0]),
        "habitable_zone_inner": weights["habitable_zone_inner"](predict.previous_velocities[1][0]),
        "habitable_zone_outer": weights["habitable_zone_outer"](predict.previous_velocities[1][0]),
        "planet_equilibrium_temperature": weights["planet_equilibrium_temperature"](predict.previous_velocities[1][0], predict.previous_coordinates[1][0], predict.previous_accelerations[1][0]),
        "transit_duration": weights["transit_duration"](predict.previous_coordinates[1][0], predict.previous_coordinates[1][1], predict.previous_velocities[1][0]),
        "gravity": predict.gravity,
        "velocity_constant": predict.velocity_constant,
        "torque": predict.torque,
        "angular_momentum": predict.angular_momentum,
        "lorentz_factor": predict.lorentz_factor,
        "asteroid_mass": predict.asteroid_mass,
        "gravitational_time_dilation": predict.gravitational_time_dilation,
        "previous_coordinates": predict.previous_coordinates,
        "predicted_coordinates": predict.predicted_coordinates,
        "previous_velocities": predict.previous_velocities,
        "previous_accelerations": predict.previous_accelerations,
        "previous_jerks": predict.previous_jerks,
        "previous_snaps": predict.previous_snaps
    }

    print("Starting score calculation...")
    score = 0.0
    print("Initial score set to 0.0")
    weight = get_weights()
    print(f"Weight obtained: {weight}")
    normalized_weights = normalize_weights(weight)
    print(f"Normalized weights: {normalized_weights}")
    
    for key in results:
        correct_value = correct_values.get(key)
        result_value = results.get(key)
        print(f"Processing key: {key}, correct value: {correct_value}, result value: {result_value}")

        if key in normalized_weights:
            weight = normalized_weights[key]
            print(f"Weight for key {key}: {weight}")

            normalization_factor = (result_value ** 2 + 1) / (result_value ** 2 + 1) if result_value != 0 else 1
            print(f"Normalization factor for key {key}: {normalization_factor}")
            weighted_contribution = weight * correct_value * normalization_factor
            print(f"Weighted contribution for key {key}: {weighted_contribution}")

            score += weighted_contribution
            print(f"Updated score: {score}")

    print("Finished score calculation.")
    return score
    
    
# FOR TESTING
# if __name__ == "__main__":
#     input_value = 1.0  # Example input
#     results = synthesized_astrophysics_analysis(input_value)
#     final_score = calculate_score(results)
#     print("Final Score:", final_score)
