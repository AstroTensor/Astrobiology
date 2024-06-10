import bittensor as bt
import asyncio
import time
from astrobiology.utils.uids import get_random_uids
from astrobiology.compute_correct_values import (
    compute_correct_values,
)
from astrobiology.directional_equations import (
    schwarzschild_radius,
    planck_energy,
    hawking_temperature,
    lorentz_factor,
    torque_equation,
    angular_momentum_equation,
    gravitational_time_dilation,
    escape_velocity,
    specific_angular_momentum,
    gravitational_wave_strain,
    flux_density,
    kepler_third_law,
    blackbody_spectrum,
    random_mass,
    random_velocity,
    random_radius,
    random_distance,
    random_frequency,
    random_chirp_mass,
)
from astrobiology.constants import G, c, M_sun
import numpy as np
from typing import List, Tuple
from astrobiology.constants import G, M_sun, c
from astrobiology.reward import calculate_score
from astrobiology.protocol import Predict
# from neurons import Validator

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

def create_predict_class(self):
    """
    Create the Predict class dynamically using equations from equations.py.
    """
    def generate_mass():
        """
        Generate a mass value for an asteroid (in kg) within a plausible range.
        """
        return np.random.uniform(1e10, 1e14) 

    def generate_velocity():
        """
        Generate a velocity value for an asteroid (in m/s) within a plausible range.
        """
        return np.random.uniform(1e3, 1e5) 

    def generate_radius():
        """
        Generate a radius value for an asteroid (in meters) within a plausible range.
        """
        return np.random.uniform(1e3, 1e6) 

    def generate_distance():
        """
        Generate a distance value for an asteroid (in meters) within a plausible range.
        """
        return np.random.uniform(1e7, 1e9) 

    def generate_frequency():
        """
        Generate a frequency value for radiation (in Hz) within a plausible range.
        """
        return np.random.uniform(1e12, 1e16) 

    def generate_chirp_mass():
        """
        Generate a chirp mass value for a binary system (in kg) within a plausible range.
        """
        return np.random.uniform(1e9, 1e11) 

    def generate_temperature():
        """
        Generate a temperature value for the asteroid (in K) within a plausible range.
        """
        return np.random.uniform(100, 1000) 

    def generate_luminosity():
        """
        Generate a luminosity value for the asteroid (in W) within a plausible range.
        """
        return np.random.uniform(1e20, 1e30)

    def generate_time():
        """
        Generate a time value for the asteroid's lifecycle (in seconds) within a plausible range.
        """
        return np.random.uniform(1e5, 1e9)

    mass = generate_mass()
    velocity = generate_velocity()
    radius = generate_radius()
    distance = generate_distance()
    frequency = generate_frequency()
    chirp_mass = generate_chirp_mass()
    temperature = generate_temperature()
    luminosity = generate_luminosity()
    time = generate_time()
    vali_dict = {"weight": self.wallet}


    gravity = G * mass / radius**2
    velocity_constant = velocity
    torque = torque_equation(mass, radius)
    angular_momentum = angular_momentum_equation(mass, velocity, radius)
    lorentz = lorentz_factor(velocity)
    time_dilation = gravitational_time_dilation(mass, radius)
    escape_vel = escape_velocity(mass, radius)
    specific_ang_mom = specific_angular_momentum(radius, velocity)
    grav_wave_strain = gravitational_wave_strain(distance, chirp_mass)
    flux = flux_density(planck_energy(frequency), distance)
    kepler_mass = kepler_third_law(radius, escape_vel)
    blackbody_flux = blackbody_spectrum(frequency, mass)
    previous_coordinates = [(0, 0, 0), (radius, radius, radius)]
    previous_velocities = [(0, 0, 0), (velocity, velocity, velocity)]
    previous_accelerations = [(0, 0, 0), (gravity, gravity, gravity)]
    previous_jerks = [(0, 0, 0), (torque, torque, torque)]
    previous_snaps = [(0, 0, 0), (lorentz, lorentz, lorentz)]

    predict_instance = Predict(
        gravity=gravity,
        velocity_constant=velocity_constant,
        torque=torque,
        angular_momentum=angular_momentum,
        lorentz_factor=lorentz,
        asteroid_mass=mass,
        gravitational_time_dilation=time_dilation,
        radius=radius,
        temperature=temperature,
        luminosity=luminosity,
        time=time,
        previous_coordinates=previous_coordinates,
        previous_velocities=previous_velocities,
        previous_accelerations=previous_accelerations,
        previous_jerks=previous_jerks,
        previous_snaps=previous_snaps
    )
    
    return vali_dict, predict_instance

async def forward(self):
    print("Starting the validator forward function...")
    # Select miners to query
    # miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    # print(f"Miner UIDs selected: {miner_uids}")

    # Create the Predict instance using the dynamically created parameters
    values, predict_synapse = create_predict_class(self)
    print("Predict instance created.", predict_synapse)

    # Query the network
    responses = []
    miner_uids = list(range(len(self.metagraph.axons)))
    # print("axon:", self.metagraph.axons[miner_uids])
    responses = await self.dendrite(
        axons=[self.metagraph.axons[i] for i in miner_uids],
        synapse=predict_synapse,
        deserialize=False,
        timeout = 3
    )
    # print("responses received:", responses)
    values["grav_constant"] = 9.80665
    rewards = [0] * len(self.metagraph.axons)

    for uid, response in zip(miner_uids, responses):
        if response.prediction_dict is not None:
            bt.logging.info(f"Received response from {uid}: {response}")
        else:
            bt.logging.info(f"No response received from {uid}.")
            continue
        current_time = time.time()
        values["time"] = current_time
        values["response"] = response
        values["u"] = [uid, response.dendrite.hotkey]
        correct_values = compute_correct_values(predict_synapse, values)
        response_score = calculate_score(predict_synapse, response.prediction_dict, correct_values)
        # rescaled_scores = synthesized_astrophysics_analysis(response_scores)
        print(f"Response score calculated: {response_score}")
        rewards[uid] = response_score
    
    rewards = [max(0, reward) for reward in rewards]
    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards
    print("updating scores...")
    self.update_scores(rewards, miner_uids)

# # TESTS
# if __name__ == "__main__":
#     # vali = Validator()
#     print("Creating Predict instance...")
#     predict_instance = create_predict_class()
#     print("Predict instance created:", predict_instance)
    
#     print("Running forward function...")
#     asyncio.run(forward(Validator))
#     print("Finished running forward function.")
