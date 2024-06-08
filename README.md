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

Miners are passed a stream of asteroid flight path information from Validators and must return predictions, the coordinates of the moving objects after some time interval. You can see and run the base miner in `neurons/miner.py` although we recommend that miners fill in the blanks to do even better on the subnet.

---

## Validating

Validators reward miners based on the Euclidean distance between the predicted location of the asteroid at some later date and the coordinates returned by the miners. The closer the miners are to the predicted coordinates, the better they fare in the reward mechanism and are paid higher. The raw asteroid location data is attained from our endpoint as the ground truth although we plan on opening this up so that any validator can directly access the data at a later date. To run the validator simply call execute `neurons/validator.py`.

---

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
