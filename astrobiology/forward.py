import numpy as np
from typing import List, Tuple
from utils.equations import schwarzschild_radius, planck_energy, hawking_temperature, some_other_complex_equation
from utils.constants import G, M_sun, c
from reward import calculate_score
from protocol import Predict

def compute_transformed_value_1(predict: Predict) -> float:
    return schwarzschild_radius(predict.asteroid_mass)

def compute_transformed_value_2(predict: Predict) -> float:
    return planck_energy(predict.velocity_constant)

def compute_transformed_value_3(predict: Predict) -> float:
    return hawking_temperature(predict.asteroid_mass)

def compute_detected_peaks(predict: Predict) -> int:
    return len(predict.previous_coordinates)

def compute_strain_amplitude(predict: Predict) -> float:
    return predict.velocity_constant * predict.gravity / predict.asteroid_mass

def compute_total_energy(predict: Predict) -> float:
    return 0.5 * predict.asteroid_mass * (predict.velocity_constant**2)

def compute_main_sequence_lifetime(predict: Predict) -> float:
    return predict.asteroid_mass * 1e10

def compute_white_dwarf_radius(predict: Predict) -> float:
    return predict.asteroid_mass * 7000 / M_sun

def compute_neutron_star_radius(predict: Predict) -> float:
    return predict.asteroid_mass * 10 / M_sun

def compute_luminosity(predict: Predict) -> float:
    return predict.asteroid_mass / M_sun

def compute_supernova_energy(predict: Predict) -> float:
    return predict.asteroid_mass * 1e44 / M_sun

def compute_final_core_mass(predict: Predict) -> float:
    return predict.asteroid_mass / 2

def compute_final_envelope_mass(predict: Predict) -> float:
    return predict.asteroid_mass / 10

def compute_planck_spectrum(predict: Predict) -> float:
    return predict.gravity * 1e-18 / predict.velocity_constant

def compute_cmb_power_spectrum(predict: Predict) -> float:
    return predict.gravity * 1e-9 / predict.velocity_constant

def compute_angular_diameter_distance(predict: Predict) -> float:
    return predict.asteroid_mass * 1e3 / M_sun

def compute_sound_horizon(predict: Predict) -> float:
    return predict.asteroid_mass * 1e2 / M_sun

def compute_reionization_history(predict: Predict) -> float:
    return predict.lorentz_factor * 0.5

def compute_dark_matter_density_profile(predict: Predict) -> float:
    return predict.gravity * 0.3 / predict.velocity_constant

def compute_rotation_curve_velocity(predict: Predict) -> float:
    return predict.velocity_constant * 200 / predict.gravity

def compute_dark_matter_mass_within_radius(predict: Predict) -> float:
    return predict.asteroid_mass * 1e12 / M_sun

def compute_lensing_deflection_angle(predict: Predict) -> float:
    return predict.gravity * 1.0 / predict.velocity_constant

def compute_transit_depth(predict: Predict) -> float:
    return predict.asteroid_mass * 0.01 / M_sun

def compute_radial_velocity_amplitude(predict: Predict) -> float:
    return predict.velocity_constant * 10 / predict.gravity

def compute_habitable_zone_inner(predict: Predict) -> float:
    return 0.95

def compute_habitable_zone_outer(predict: Predict) -> float:
    return 1.37

def compute_planet_equilibrium_temperature(predict: Predict) -> float:
    return 288

def compute_transit_duration(predict: Predict) -> float:
    return 0.5

def compute_correct_values(predict: Predict) -> dict:
    """
    Compute the correct values for reward calculation based on the current state of the asteroid.
    Returns:
        dict: A dictionary of computed correct values.
    """
    correct_values = {
        "transformed_value_1": compute_transformed_value_1(predict),
        "transformed_value_2": compute_transformed_value_2(predict),
        "transformed_value_3": compute_transformed_value_3(predict),
        "detected_peaks": compute_detected_peaks(predict),
        "strain_amplitude": compute_strain_amplitude(predict),
        "total_energy": compute_total_energy(predict),
        "main_sequence_lifetime": compute_main_sequence_lifetime(predict),
        "white_dwarf_radius": compute_white_dwarf_radius(predict),
        "neutron_star_radius": compute_neutron_star_radius(predict),
        "luminosity": compute_luminosity(predict),
        "supernova_energy": compute_supernova_energy(predict),
        "final_core_mass": compute_final_core_mass(predict),
        "final_envelope_mass": compute_final_envelope_mass(predict),
        "planck_spectrum": compute_planck_spectrum(predict),
        "cmb_power_spectrum": compute_cmb_power_spectrum(predict),
        "angular_diameter_distance": compute_angular_diameter_distance(predict),
        "sound_horizon": compute_sound_horizon(predict),
        "reionization_history": compute_reionization_history(predict),
        "dark_matter_density_profile": compute_dark_matter_density_profile(predict),
        "rotation_curve_velocity": compute_rotation_curve_velocity(predict),
        "dark_matter_mass_within_radius": compute_dark_matter_mass_within_radius(predict),
        "lensing_deflection_angle": compute_lensing_deflection_angle(predict),
        "transit_depth": compute_transit_depth(predict),
        "radial_velocity_amplitude": compute_radial_velocity_amplitude(predict),
        "habitable_zone_inner": compute_habitable_zone_inner(predict),
        "habitable_zone_outer": compute_habitable_zone_outer(predict),
        "planet_equilibrium_temperature": compute_planet_equilibrium_temperature(predict),
        "transit_duration": compute_transit_duration(predict),
    }

    return correct_values

def compute_score(predict: Predict) -> float:
    """
    Compute the score for the current state of the asteroid.
    Returns:
        float: The computed score.
    """
    correct_values = compute_correct_values(predict)
    return calculate_score(correct_values)

# FOR TESTING
# if __name__ == "__main__":
#     predict_instance = Predict(
#         gravity=9.8,
#         velocity_constant=30000,
#         torque=1e5,
#         angular_momentum=1e6,
#         lorentz_factor=1.01,
#         asteroid_mass=1e12,
#         gravitational_time_dilation=1.0001,
#         previous_coordinates=[(0, 0, 0), (1, 1, 1)],
#         previous_velocities=[(0, 0, 0), (1, 1, 1)],
#         previous_accelerations=[(0, 0, 0), (1, 1, 1)],
#         previous_jerks=[(0, 0, 0), (1, 1, 1)],
#         previous_snaps=[(0, 0, 0), (1, 1, 1)]
#     )

#     score = compute_score(predict_instance)
#     print("Computed Score:", score)
