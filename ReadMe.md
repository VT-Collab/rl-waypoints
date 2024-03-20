# Waypoint-Based Reinforcement Learning for Robot Manipulation Tasks

This repository provides our implementation of Waypoint-Based RL. The videos for experiments can be found [here](https://youtu.be/MMEd-lYfq4Y)

## Installation
Clone the repository using 
```bash
git clone https://github.com/VT-Collab/rl-waypoints.git
```

## Implementation of rl-waypoints
Navigate to the rl-waypoints repository using 
```bash
cd rl-waypoints
cd rlwp
```

To train the robot for manipulation tasks in the Robosuite simulation environment, run the following commands
```bash
python3 main.py train=True task=<task_name>
```
The complete set of arguments can be found in the \cfg folder. The tasks for the training can be from the following: {Lift, Stack, NutAssembly, PickPlace, Door}

## Testing the trained models
We provide a trained model for each task for our approach. The trained models can be evaluated as follows
```bash
python3 main.py test=True task=<task_name> run_name=test render=True
```