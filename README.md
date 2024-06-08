<div align="center">
  <img src="https://github.com/AstroTensor/Astrobiology/raw/main/astrobiology.jpeg" alt="Astrobiology" width="600">
  
  # **Astrobiology Subnet**
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
- [File Descriptions](#file-descriptions)
- [Neural Network Architecture](#neural-network-architecture)
- [Holistic Scoring and Reward Mechanism](#holistic-scoring-and-reward-mechanism)

---

## Introduction

Predicting the path of asteroids is both a computationally difficult challenge and one of incredible importance for safeguarding humanity against stray, interplanetary objects. This subnet uses Bittensor's incentivization system to reward miners for making accurate predictions about the flight paths of asteroids. Because of the complex interaction between multiple moving bodies in space, miners must employ complex prediction methods and artificial intelligence to compute the likely flight paths of the moving objects -- and potentially save humanity.

To read more about the research behind this method see:

Asteroid Impact Predictions
https://www.researchgate.net/publication/299723990_Asteroid_Impact_Predictions: \n

Identifying Earth-impacting asteroids using an artificial neural network:\n
https://www.aanda.org/articles/aa/full_html/2020/02/aa35983-19/aa35983-19.html

A probabilistic asteroid impact risk model: assessment of sub-300 m impacts: \n
https://www.sciencedirect.com/science/article/pii/S0019103516307126

---

## Mining

Miners receive a stream of asteroid flight path data from Validators and are tasked with returning predictions, specifically the coordinates of the moving objects after a certain time interval. The base miner implementation can be found in `neurons/miner.py`, but miners are encouraged to enhance this to improve performance on the subnet.

### Neural Network Architecture

The mining process utilizes a sophisticated neural network architecture designed to compute "correct scores" based on the asteroid flight path data. This network is tailored to manage the multi-dimensional, temporal-spatial complexities of the data to deliver highly accurate predictions.

#### Setting Up the Neural Network

Miners should set up their neural networks using a deep learning framework like TensorFlow or PyTorch. The network configuration should include:

- **Input Handling:** Process multi-dimensional time-series data including asteroid coordinates, velocities, accelerations, jerks, and snaps.
- **Feature Engineering:** Implement advanced techniques such as convolutional layers, recurrent layers, and attention mechanisms to extract high-dimensional features and capture complex patterns in the data.
- **Temporal Dynamics:** Incorporate LSTM or GRU cells to model the temporal dependencies, enabling the network to learn the progression of the asteroid's trajectory over time.
- **Optimization Algorithms:** Use advanced optimization algorithms like AdamW, Ranger, or Lookahead to minimize the loss function, which should ideally combine Mean Squared Error with custom loss functions that penalize larger deviations.
- **Hyperparameter Tuning:** Apply techniques such as Bayesian Optimization, Hyperband, or Genetic Algorithms to fine-tune the model parameters for optimal performance.

---

## Validating

Validators assess and reward miners based on the accuracy of their predictions. The Euclidean distance between the predicted and actual coordinates of the asteroid at a later date determines the reward; closer predictions yield higher rewards. The raw asteroid location data is sourced from our endpoint as the ground truth, with plans to make this data directly accessible to validators in the future.

### Holistic Scoring and Reward Mechanism

The validator uses a comprehensive scoring and reward system involving several detailed steps:

- **Data Acquisition:** Continuously fetch raw asteroid location data from a secure endpoint to ensure the ground truth data is accurate and up-to-date.

  ```latex
  \begin{equation}
  \text{Data}_{\text{raw}} = \int_{t_0}^{t_n} \text{fetch}(\text{endpoint})
  \end{equation}
  ```

- **Correct Values Calculation:** Dynamically compute the "correct values" for the asteroid's future coordinates using the `forward.py` module, which leverages complex astrophysical equations and constants from `utils.equations.py` and `utils.constants.py`.

  ```latex
  \begin{equation}
  \text{Correct}_{\text{values}} = f(\text{astrophysical\_equations}, \text{constants})
  \end{equation}
  ```

- **Prediction Evaluation:** Calculate the Euclidean distance between the predicted coordinates and the correct values in a multi-dimensional space, considering the temporal-spatial intricacies of the data.

  ```latex
  \begin{equation}
  \text{Distance} = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
  \end{equation}
  ```

- **Score Calculation:** Use a sophisticated scoring function defined in `reward.py`, applying different weights to each input and utilizing a series of helper functions to compute the normalized differences. The weighted sum of these differences determines the final score.

  ```latex
  \begin{equation}
  \text{Score} = \sum_{i=1}^{n} w_i \cdot \left( \frac{x_i - y_i}{\text{max}(x_i, y_i)} \right)
  \end{equation}
  ```

- **Reward Allocation:** Distribute rewards to miners based on their scores using a non-linear reward mechanism, which offers higher rewards for exceptionally close predictions. This distribution considers the overall performance of the miners to ensure a fair and balanced incentive structure.

  ```latex
  \begin{equation}
  \text{Reward} = \frac{1}{1 + e^{-\text{Score}}}
  \end{equation}
  ```

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

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
