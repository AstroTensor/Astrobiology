# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

from typing import List, Tuple
import bittensor as bt

class Predict(bt.Synapse):
    """
    This class represents a predictive model that utilizes the previous coordinates of an asteroid
    to compute its future trajectory. The model takes into account various physical parameters such
    as gravity, velocity, torque, angular momentum, and relativistic effects to enhance the accuracy of the prediction.
    """

    # Gravitational constant affecting the asteroid (m/s^2)
    gravity: float = None

    # Constant velocity of the asteroid in space (m/s)
    velocity_constant: float = None

    # Torque affecting the asteroid's rotational motion (N·m)
    torque: float = None

    # Angular momentum of the asteroid (kg·m^2/s)
    angular_momentum: float = None

    # Relativistic factor to account for velocities approaching the speed of light, dimensionless
    lorentz_factor: float = None

    # Mass of the asteroid (kg)
    asteroid_mass: float = None

    # Time dilation factor due to gravitational time dilation, dimensionless
    gravitational_time_dilation: float = None

    # List of tuples representing the asteroid's previous coordinates in a 3D space (x, y, z)
    previous_coordinates: List[Tuple[float, float, float]]

    # List of tuples predicting the asteroid's coordinates in the next 5 minutes (x, y, z)
    predicted_coordinates: List[Tuple[float, float, float]] = None

    # List of tuples representing the asteroid's velocity vectors at previous coordinates (vx, vy, vz)
    previous_velocities: List[Tuple[float, float, float]] = None

    # List of tuples representing the asteroid's acceleration vectors at previous coordinates (ax, ay, az)
    previous_accelerations: List[Tuple[float, float, float]] = None

    # List of tuples representing the asteroid's jerk (rate of change of acceleration) vectors at previous coordinates (jx, jy, jz)
    previous_jerks: List[Tuple[float, float, float]] = None

    # List of tuples representing the asteroid's snap (rate of change of jerk) vectors at previous coordinates (sx, sy, sz)
    previous_snaps: List[Tuple[float, float, float]] = None