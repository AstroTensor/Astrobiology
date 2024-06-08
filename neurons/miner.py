# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Meldf

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import typing
import bittensor as bt

# Bittensor Miner Template:
import astrobiology

# import base miner class which takes care of most of the boilerplate
from astrobiology.base.miner import BaseMinerNeuron
from astrobiology.base.model import AsteroidModelPredictor
from astrobiology.base.dynamic_markov_simulation import fill_dynamic_arguments_based_on_prediction

class AstroMiner(BaseMinerNeuron):
    """
    This class, AstroMiner, inherits from BaseMinerNeuron and is specifically tailored to predict the future
    locations of asteroids based on their previous coordinates. The class encapsulates the complexities of
    astrophysical computations and serves as a conduit between raw data and actionable predictions.
    # The function `predict_trajectory` is invoked on the model, which integrates several critical parameters:
        # 1. `gravity`: This represents the gravitational constant that significantly influences the asteroid's trajectory through the classical Newtonian gravitational pull.
        # 2. `velocity_constant`: A scalar that encapsulates the constant velocity component of the asteroid, adhering to the principle of inertia that an object in motion stays in motion unless acted upon by an external force.
        # 3. `torque`: This parameter introduces the rotational dynamics affecting the asteroid, contributing to its angular momentum and thus altering its trajectory through rotational forces.
        # 4. `angular_momentum`: It quantifies the momentum associated with the asteroid's rotation, which is conserved in a closed system, thereby playing a crucial role in the prediction of its path.
        # 5. `lorentz_factor`: A relativistic correction factor that adjusts velocities approaching the speed of light, ensuring that the predictions respect the limitations imposed by the theory of relativity.
        # 6. `asteroid_mass`: The mass of the asteroid, a fundamental property that influences both gravitational interactions and inertial resistance to changes in motion.
        # 7. `gravitational_time_dilation`: This parameter accounts for the effects of time dilation due to gravity as predicted by general relativity, affecting the perception of time as influenced by the gravitational field.
        # 8. `previous_coordinates`: A historical log of the asteroid's coordinates, providing a trajectory context that helps in predicting future positions based on past movement.
        # 9. `predicted_coordinates`: Initial predictions of future coordinates that serve as a basis for iterative refinement through the model's predictive processes.
        # 10. `previous_velocities`: Historical data of how fast the asteroid was moving at each previously logged coordinate, offering insights into its dynamic state over time.
        # 11. `previous_accelerations`: This captures changes in velocity, providing a second derivative context of the motion that is crucial for understanding forces acting on the asteroid.
        # 12. `previous_jerks`: The rate of change of acceleration, a third derivative of position that provides a deeper insight into the variability of forces influencing the asteroid's trajectory.
        # This process involves:
        # 1. Lorentz Transformations: Adjusting the parameters for relativistic effects, ensuring that the predictions are accurate even at velocities approaching the speed of light.
        # 2. Tensor Calculus: Utilizing tensors to handle the complexities of multidimensional data and transformations, crucial for accurately modeling the asteroid's motion in three-dimensional space.
        # 3. Multidimensional Scaling: Scaling down the high-dimensional data to manageable forms without losing the essence of the underlying dynamics, facilitating more efficient computations.
        # 4. Quantum Mechanics Principles: Applying principles of quantum mechanics to account for subatomic effects that might influence the asteroid's trajectory at a very small scale.
        # 5. General Relativity Adjustments: Incorporating the effects of gravity as spacetime curvature, which affects the trajectory of the asteroid significantly, especially near massive bodies.
        # 6. Non-linear Dynamics: Dealing with the chaotic nature of asteroid trajectories that result from small changes in initial conditions leading to vastly different outcomes.
        # These steps collectively contribute to the generation of a set of predicted future coordinates, representing the most probable trajectory of the asteroid, given all the current and historical data available to the model.
        
    """

    def __init__(self, config=None):
        """
        The constructor of the AstroMiner class initializes the instance by setting up the configuration
        and preparing the predictive model that will be used to forecast asteroid trajectories.
        
        Args:
            config (Optional[dict]): Configuration parameters that may include settings pertinent to the
                                     model's operation such as learning rates, epochs, etc.
        """
        super(AstroMiner, self).__init__(config=config)  # Initialize the superclass with the provided configuration.
        self.model = AsteroidModelPredictor(config=config)  # Instantiate the model predictor with the same configuration.

    async def forward(
        self, synapse: astrobiology.protocol.Predict
    ) -> astrobiology.protocol.Predict:
        """
        The 'forward' method is the crux of the AstroMiner's functionality. It processes incoming synapse
        objects, which encapsulate the data necessary for making predictions about asteroid trajectories.
        This method leverages the predictive prowess of the underlying model to compute future coordinates
        based on a multitude of astrophysical parameters provided via the synapse.

        Args:
            synapse (astrobiology.protocol.Predict): A data structure containing all the necessary parameters
                                                     to predict the future trajectory of an asteroid. These
                                                     parameters include gravitational constants, velocities,
                                                     and past positional data among others.

        Returns:
            astrobiology.protocol.Predict: The synapse object is returned with the 'predicted_coordinates' field
                                           populated with the computed future locations of the asteroid, thereby
                                           providing a direct insight into the expected trajectory based on current
                                           and past physical conditions.
        """
        # The following block of code leverages the sophisticated predictive model encapsulated within the AstroMiner's architecture to compute the future trajectory of an asteroid. This computation is not merely a straightforward prediction but a complex synthesis of multiple astrophysical parameters and laws, intricately woven together through advanced mathematical formulations and machine learning techniques.

        # The function `predict_trajectory` is invoked on the model, which integrates several critical parameters:
        # 1. `gravity`: This represents the gravitational constant that significantly influences the asteroid's trajectory through the classical Newtonian gravitational pull.
        # 2. `velocity_constant`: A scalar that encapsulates the constant velocity component of the asteroid, adhering to the principle of inertia that an object in motion stays in motion unless acted upon by an external force.
        # 3. `torque`: This parameter introduces the rotational dynamics affecting the asteroid, contributing to its angular momentum and thus altering its trajectory through rotational forces.
        # 4. `angular_momentum`: It quantifies the momentum associated with the asteroid's rotation, which is conserved in a closed system, thereby playing a crucial role in the prediction of its path.
        # 5. `lorentz_factor`: A relativistic correction factor that adjusts velocities approaching the speed of light, ensuring that the predictions respect the limitations imposed by the theory of relativity.
        # 6. `asteroid_mass`: The mass of the asteroid, a fundamental property that influences both gravitational interactions and inertial resistance to changes in motion.
        # 7. `gravitational_time_dilation`: This parameter accounts for the effects of time dilation due to gravity as predicted by general relativity, affecting the perception of time as influenced by the gravitational field.
        # 8. `previous_coordinates`: A historical log of the asteroid's coordinates, providing a trajectory context that helps in predicting future positions based on past movement.
        # 9. `predicted_coordinates`: Initial predictions of future coordinates that serve as a basis for iterative refinement through the model's predictive processes.
        # 10. `previous_velocities`: Historical data of how fast the asteroid was moving at each previously logged coordinate, offering insights into its dynamic state over time.
        # 11. `previous_accelerations`: This captures changes in velocity, providing a second derivative context of the motion that is crucial for understanding forces acting on the asteroid.
        # 12. `previous_jerks`: The rate of change of acceleration, a third derivative of position that provides a deeper insight into the variability of forces influencing the asteroid's trajectory.
        # This process involves:
        # 1. Lorentz Transformations: Adjusting the parameters for relativistic effects, ensuring that the predictions are accurate even at velocities approaching the speed of light.
        # 2. Tensor Calculus: Utilizing tensors to handle the complexities of multidimensional data and transformations, crucial for accurately modeling the asteroid's motion in three-dimensional space.
        # 3. Multidimensional Scaling: Scaling down the high-dimensional data to manageable forms without losing the essence of the underlying dynamics, facilitating more efficient computations.
        # 4. Quantum Mechanics Principles: Applying principles of quantum mechanics to account for subatomic effects that might influence the asteroid's trajectory at a very small scale.
        # 5. General Relativity Adjustments: Incorporating the effects of gravity as spacetime curvature, which affects the trajectory of the asteroid significantly, especially near massive bodies.
        # 6. Non-linear Dynamics: Dealing with the chaotic nature of asteroid trajectories that result from small changes in initial conditions leading to vastly different outcomes.
        # These steps collectively contribute to the generation of a set of predicted future coordinates, representing the most probable trajectory of the asteroid, given all the current and historical data available to the model.
        try:
            predicted_coordinates = self.model.predict_trajectory(
                gravity=synapse.gravity,  # Gravitational constant affecting the asteroid.
                velocity_constant=synapse.velocity_constant,  # Constant velocity of the asteroid in space.
                torque=synapse.torque,  # Torque affecting the asteroid's rotational motion.
                angular_momentum=synapse.angular_momentum,  # Angular momentum of the asteroid.
                lorentz_factor=synapse.lorentz_factor,  # Relativistic factor for velocities approaching the speed of light.
                asteroid_mass=synapse.asteroid_mass,  # Mass of the asteroid.
                gravitational_time_dilation=synapse.gravitational_time_dilation,  # Time dilation factor due to gravity.
                previous_coordinates=synapse.previous_coordinates,  # List of previous coordinates of the asteroid.
                previous_velocities=synapse.previous_velocities,  # List of previous velocities of the asteroid.
                previous_accelerations=synapse.previous_accelerations,  # List of previous accelerations.
                previous_jerks=synapse.previous_jerks  # List of previous jerks (rate of change of acceleration).
            )
            synapse.predicted_coordinates = predicted_coordinates  # Update the synapse with the new predicted coordinates.
            synapse = fill_dynamic_arguments_based_on_prediction( predicted_coordinates, synapse ) # Update Markov terms.
        except:
            pass
        return synapse  # Return the updated synapse object, now containing the future trajectory predictions.

    async def blacklist(
        self, synapse: astrobiology.protocol.Predict
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: astrobiology.protocol.Predict) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0
        
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with AstroMiner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
