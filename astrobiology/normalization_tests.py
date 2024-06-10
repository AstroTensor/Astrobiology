import numpy as np

def normalize_weights(weight_values):
    # Step 1: Normalize the weights
    total = sum(weight_values.values())
    if total == 0:
        raise ValueError("Total sum of weights is zero, cannot normalize.")
    
    normalized_weights = {key: (0.0 if not np.isfinite(np.float64(value)) else value / total) for key, value in weight_values.items()}
    
    print("After initial normalization:")
    for key, value in normalized_weights.items():
        print(f"{key}: {value}")

    # Step 2: Check if any weight exceeds 0.5 and cap it
    max_allowed_weight = 0.5
    capped_weights = {}
    uncapped_sum = 0.0
    for key, value in normalized_weights.items():
        if value > max_allowed_weight:
            capped_weights[key] = max_allowed_weight
        else:
            uncapped_sum += value
            capped_weights[key] = value

    # Step 3: Calculate the total weight after capping
    capped_total = sum(capped_weights.values())

    # Step 4: Re-normalize the remaining weights if needed
    if capped_total != 1.0:
        scale_factor = (1.0 - sum(value for value in capped_weights.values() if value == max_allowed_weight)) / uncapped_sum
        for key, value in capped_weights.items():
            if value < max_allowed_weight:
                capped_weights[key] = value * scale_factor

    print("After re-normalization:")
    for key, value in capped_weights.items():
        print(f"{key}: {value}")

    return capped_weights

weight_values = {
    'schwarzschild_radius': np.random.rand(),
    'planck_energy': np.random.rand(),
    'hawking_temperature': np.random.rand(),
    'detected_peaks': np.random.rand(),
    'strain_amplitude': np.random.rand(),
    'total_energy': np.random.rand(),
    'main_sequence_lifetime': np.random.rand(),
    'white_dwarf_radius': np.random.rand(),
    'neutron_star_radius': np.random.rand(),
    'luminosity': np.random.rand(),
    'supernova_energy': np.random.rand(),
    'final_core_mass': np.random.rand(),
    'final_envelope_mass': np.random.rand(),
    'planck_spectrum': np.random.rand(),
    'cmb_power_spectrum': np.random.rand(),
    'angular_diameter_distance': np.random.rand(),
    'sound_horizon': np.random.rand(),
    'reionization_history': np.random.rand(),
    'dark_matter_density_profile': np.random.rand(),
    'rotation_curve_velocity': np.random.rand(),
    'dark_matter_mass_within_radius': np.random.rand(),
    'lensing_deflection_angle': np.random.rand(),
    'transit_depth': np.random.rand(),
    'radial_velocity_amplitude': np.random.rand(),
    'habitable_zone_inner': np.random.rand(),
    'habitable_zone_outer': np.random.rand(),
    'planet_equilibrium_temperature': np.random.rand(),
    'transit_duration': np.random.rand()
}

normalized_weights = normalize_weights(weight_values)
print("Final normalized weights:")
print(normalized_weights)
