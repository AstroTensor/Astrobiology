<div align="center">
  <img src="https://github.com/AstroTensor/Astrobiology/raw/main/astrobiology.jpeg" alt="Astrobiology" width="600">
  
  # **Astrobiology Subnet**
  [![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

  ---
  [Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---

## Table of Contents
- [Introduction](#introduction)
- [Mining](#mining)
- [Validating](#validating)
- [Installation](#installation)
  - [Before you proceed](#before-you-proceed)
  - [Install](#install)
- [License](#license)

---

## Introduction

Predicting the path of asteroids is both a computationally difficult challenge and one of incredible importance for safeguarding humanity against stray, interplanetary objects. This subnet uses Bittensor's incentivization system to reward miners for making accurate predictions about the flight paths of asteroids. Because of the complex interaction between multiple moving bodies in space, miners must employ complex prediction methods and artificial intelligence to compute the likely flight paths of the moving objects -- and potentially save humanity.

---

## Mining

Miners are passed a stream of asteroid flight path information from Validators and must return predictions, the coordinates of the moving objects after some time interval. You can see and run the base miner in neurons/miner.py although we recommend that miners fill in the blanks to do even better on the subnet.

The miner utilizes a highly convoluted neural network architecture to solve for the "correct scores" using the asteroid flight path information. The neural network is designed to handle the multi-dimensional, temporal-spatial complexities of the data and provide highly accurate predictions.

Setting Up the Neural Network
Miners are encouraged to set up their neural networks using a deep learning framework of their choice, such as TensorFlow or PyTorch. The network should be designed to:

Input Handling: Process multi-dimensional time-series data of asteroid coordinates, velocities, accelerations, jerks, and snaps.
Feature Engineering: Extract high-dimensional features using advanced techniques like convolutional layers, recurrent layers, and attention mechanisms to capture the intricate patterns in the data.
Temporal Dynamics: Model the temporal dependencies using LSTM or GRU cells, ensuring the network can learn the progression of the asteroid's trajectory over time.
Optimization Algorithms: Employ sophisticated optimization algorithms such as AdamW, Ranger, or Lookahead to minimize the loss function, which should ideally be a combination of Mean Squared Error and custom loss functions designed to penalize larger deviations.
Hyperparameter Tuning: Utilize hyperparameter optimization techniques like Bayesian Optimization, Hyperband, or Genetic Algorithms to fine-tune the model parameters for optimal performance.

---

## Validating

Validators reward miners based on the Euclidean distance between the predicted location of the asteroid at some later date and the coordinates returned by the miners. The closer the miners are to the predicted coordinates, the better they fare in the reward mechanism and are paid higher. The raw asteroid location data is attained from our endpoint as the ground truth although we plan on opening this up so that any validator can directly access the data at a later date.

Holistic Scoring and Reward Mechanism
The validator employs a multifaceted approach to scoring and rewarding the miners. The process involves the following intricate steps:

Data Acquisition: Raw asteroid location data is continuously fetched from a secure endpoint, ensuring the ground truth data is up-to-date and accurate.

Correct Values Calculation: Using the forward.py module, the validator dynamically computes the "correct values" for the asteroid's future coordinates. This involves leveraging a suite of complex astrophysical equations and constants from utils.equations.py and utils.constants.py.

Prediction Evaluation: The validator assesses the miners' predictions by calculating the Euclidean distance between the predicted coordinates and the correct values. This calculation is performed in a multi-dimensional space, taking into account the intricacies of temporal-spatial data.

Score Calculation: The score is computed using a sophisticated scoring function defined in reward.py. This function applies different weights to each input and uses a series of helper functions to compute the normalized differences. The weighted sum of these differences determines the final score.

Reward Allocation: The validators allocate rewards to miners based on their scores. The reward mechanism is designed to be non-linear, with higher rewards for predictions that are exceptionally close to the correct values. The reward distribution takes into account the overall performance of the miners, ensuring a fair and balanced incentive structure.

---


## File descriptions

directional_equations.py
Defines essential astrophysical equations and constants fundamental for various calculations in astrophysics.

gravitational_wave_analysis.py
Provides tools for detecting and analyzing gravitational waves.

stellar_evolution.py
Focuses on the lifecycle of stars, providing functions to estimate various properties at different evolutionary stages.

cosmic_microwave_background_analysis.py
Provides tools for analyzing the Cosmic Microwave Background (CMB), a critical aspect of understanding the early universe.

dark_matter_analysis.py
Focuses on the study and analysis of dark matter, providing functions to calculate density profiles, rotational velocities, mass distributions, and gravitational lensing effects.

exoplanet_detection.py
Provides tools for detecting and analyzing exoplanets using various astrophysical techniques.

reward.py
Defines a scoring function that evaluates the accuracy of astrophysical predictions by comparing results against dynamically computed "correct values" and applying different weights.

forward.py
Integrates values from the Predict class with astrophysical equations from utils.equations and constants from utils.constants. It dynamically computes "correct values" and calculates a final score using the calculate_score function from reward.py.

## Installation

### Before you proceed

Before you proceed with the installation of the subnet, note the following:

- Use these instructions to run your subnet locally for your development and testing, or on Bittensor testnet or on Bittensor mainnet.
- **IMPORTANT**: We **strongly recommend** that you first run your subnet locally and complete your development and testing before running the subnet on Bittensor testnet. Furthermore, make sure that you next run your subnet on Bittensor testnet before running it on the Bittensor mainnet.
- You can run your subnet either as a subnet owner, or as a subnet validator or as a subnet miner.
- **IMPORTANT**: Make sure you are aware of the minimum compute requirements for your subnet. See the [Minimum compute YAML configuration](./min_compute.yml).
- Note that installation instructions differ based on your situation: For example, installing for local development and testing will require a few additional steps compared to installing for testnet. Similarly, installation instructions differ for a subnet owner vs a validator or a miner.

### Install

```bash
python3.11 -m pip install -e .
```


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
