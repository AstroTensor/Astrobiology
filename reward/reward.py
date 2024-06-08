import numpy as np
from astrophysics_synthesis import synthesized_astrophysics_analysis

# Define weights for each result
weights = {
    "transformed_value_1": 1.0,
    "transformed_value_2": 1.0,
    "transformed_value_3": 1.0,
    "detected_peaks": 2.0,
    "strain_amplitude": 1.5,
    "total_energy": 1.2,
    "main_sequence_lifetime": 2.0,
    "white_dwarf_radius": 1.0,
    "neutron_star_radius": 1.0,
    "luminosity": 1.5,
    "supernova_energy": 2.5,
    "final_core_mass": 1.5,
    "final_envelope_mass": 1.5,
    "planck_spectrum": 1.0,
    "cmb_power_spectrum": 1.2,
    "angular_diameter_distance": 2.0,
    "sound_horizon": 2.0,
    "reionization_history": 1.0,
    "dark_matter_density_profile": 1.0,
    "rotation_curve_velocity": 1.2,
    "dark_matter_mass_within_radius": 2.0,
    "lensing_deflection_angle": 1.0,
    "transit_depth": 1.5,
    "radial_velocity_amplitude": 1.5,
    "habitable_zone_inner": 1.0,
    "habitable_zone_outer": 1.0,
    "planet_equilibrium_temperature": 1.0,
    "transit_duration": 1.5,
}

def calculate_score(inputs, correct_values):
    """
    Calculate the score based on the results and correct values using weighted differences.
    
    Parameters:
    results (dict): Dictionary containing the results from synthesized astrophysics analysis.
    
    Returns:
    float: Final computed score.
    """
    score = 0.0
    
    for key in results:
        correct_value = correct_values[key]
        weight = weights[key]
        result_value = results[key]
        
        # Normalize the result to handle different magnitudes
        normalized_difference = np.abs((result_value - correct_value) / correct_value)
        
        # Weighted contribution to the score
        weighted_contribution = weight * normalized_difference
        
        score += weighted_contribution
    
    # Inverse the score to make a higher score better (optional)
    final_score = 1 / (1 + score)
    
    return final_score

# FOR TESTING
# if __name__ == "__main__":
#     input_value = 1.0  # Example input
#     results = synthesized_astrophysics_analysis(input_value)
#     final_score = calculate_score(results)
#     print("Final Score:", final_score)
