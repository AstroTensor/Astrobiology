import bittensor as bt

from predict import Predict
from reward import calculate_score
from utils.uids import get_random_uids
from utils.equations import (
    schwarzschild_radius,
    planck_energy,
    hawking_temperature,
    gravitational_force,
    lorentz_factor,
    torque_equation,
    angular_momentum_equation,
    gravitational_time_dilation,
    random_equation_1,
    random_equation_2,
)
from utils.constants import G, c, M_sun

def create_predict_class():
    """
    Create the Predict class dynamically using equations from equations.py
    """

    # Example parameters calculated using equations from equations.py
    mass = 1e12  # Example mass of the asteroid in kg
    velocity = 30000  # Example constant velocity in m/s
    radius = 1e5  # Example radius in meters

    gravity = gravitational_force(mass, radius)
    velocity_constant = velocity
    torque = torque_equation(mass, radius)
    angular_momentum = angular_momentum_equation(mass, velocity, radius)
    lorentz = lorentz_factor(velocity)
    time_dilation = gravitational_time_dilation(mass, radius)

    # Generate dummy previous state data (you can replace it with real data)
    previous_coordinates = [(0, 0, 0), (radius, radius, radius)]
    previous_velocities = [(0, 0, 0), (velocity, velocity, velocity)]
    previous_accelerations = [(0, 0, 0), (random_equation_1(mass, radius), random_equation_1(mass, radius), random_equation_1(mass, radius))]
    previous_jerks = [(0, 0, 0), (random_equation_2(mass, velocity), random_equation_2(mass, velocity), random_equation_2(mass, velocity))]
    previous_snaps = [(0, 0, 0), (random_equation_1(velocity, radius), random_equation_1(velocity, radius), random_equation_1(velocity, radius))]

    predict_instance = Predict(
        gravity=gravity,
        velocity_constant=velocity_constant,
        torque=torque,
        angular_momentum=angular_momentum,
        lorentz_factor=lorentz,
        asteroid_mass=mass,
        gravitational_time_dilation=time_dilation,
        previous_coordinates=previous_coordinates,
        previous_velocities=previous_velocities,
        previous_accelerations=previous_accelerations,
        previous_jerks=previous_jerks,
        previous_snaps=previous_snaps
    )

    return predict_instance

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # Select miners to query
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    # Create the Predict instance using the dynamically created parameters
    predict_synapse = create_predict_class()

    # Query the network
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=predict_synapse,
        deserialize=True,
    )

    # Log the results for monitoring purposes
    bt.logging.info(f"Received responses: {responses}")

    # Define how the validator scores responses
    rewards = []
    for response in responses:
        correct_values = predict_synapse.compute_correct_values()
        response_score = calculate_score(correct_values)
        rewards.append(response_score)

    bt.logging.info(f"Scored responses: {rewards}")

    # Update the scores based on the rewards
    self.update_scores(rewards, miner_uids)

# TESTS
# if __name__ == "__main__":
#     predict_instance = create_predict_class()
#     score = compute_score(predict_instance)
#     print("Computed Score:", score)