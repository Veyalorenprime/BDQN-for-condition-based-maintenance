# DBQN-for-condition-based-maintenance
Deep Reinforcement Learning for condition based maintenance: maintenance of wind turbines as example


# Description of the project

As the project is part of the Research Chair in Risk and Resilience of Complex Systems of complex systems, we are interested in problems inspired by the use cases of the industrial partners of the chair (EDF, Orange, SNCF). In particular, we study the optimization of maintenance policies for distributed systems, in this case of a fleet of wind turbines.

A distributed system is characterized by: a system composed of many sub-components; we can speak of units or items, the units degrade over time independently and the system performance is described by the revenue given by the sum of the individual unit contributions, which consider the electricity produced, and the maintenance cost.

We need to find a coordinated maintenance maintenance strategy. This is necessary as soon as the problem has economic dependencies, which can take the form of coupling constraints (e.g. a limited number of technicians to perform maintenance) or system costs (e.g. we pay a deployment cost each time we send a team of technicians).

The degradation process is based on discrete states from which it transits stochastically given a Markov chain.

There are two types of interventions: 1) preventive operations, which intervene on a unit that has not yet broken down, and 2) corrective operations, which in contrast intervene on units that have already broken down. In general, interventions of type 2) are more expensive.

Gradually, the problem became more complicated in order to gradually study how it was solved. Thus we added a restriction on a limited number of technicians for maintenance or the fixed cost paid for sending a group of technicians.



# Libraries Used in this Project

This project utilizes the following libraries:

## NumPy
NumPy is a popular Python library used for numerical computing. It provides a powerful array manipulation capability, and allows for efficient processing of large datasets. In this project, NumPy was used extensively for data preprocessing and manipulation.

## TensorFlow
TensorFlow is a powerful open-source machine learning library developed by Google Brain team. It provides a flexible architecture for building machine learning models and supports both CPU and GPU computations. In this project, TensorFlow was used for building and training machine learning models.

## TensorFlow Keras Layers
Keras is a high-level API for building and training deep learning models. It is included in TensorFlow as a set of pre-built functions, and provides an easy-to-use interface for building complex models. In this project, we used TensorFlow Keras Layers for defining the layers of our neural network model.

All of these libraries were instrumental in the development of this project, and their functionality and ease of use allowed for efficient and effective implementation of our models.





# Project Structure

This project consists of two main directories: `agents` and `env`. Each of these directories contains a set of files that implement the agent and the environment, respectively.

## Agents
The `agents` directory contains the implementation of the reinforcement learning agent. In particular, the file `Agent_BDQN.py` implements the `BDQN_Agent` class, which defines the behavior and learning strategy of the agent. This class makes use of the TensorFlow and NumPy libraries to build and train a deep Q-network (DQN) for solving the air turbines problem.

## Environment
The `env` directory contains the implementation of the environment in which the agent operates. This environment is designed to simulate the behavior of a set of air turbines, and is composed of several files:

- `environnement.py`: This file contains the `Environnement` class, which provides an initial way of initializing the environment with a certain threshold, wearing function, and production of the air turbines.

- `items.py`: This file contains the `Item` class, which represents the air turbines in the environment.

- `states.py`: This file contains the `State` class, which is capable of processing the possible states and actions from a given state.

- `utils_action.py`: This file contains various auxiliary functions used in the environment.

- `wearing_functions.py`: This file contains the functions that permit the representation of the Markov chain of the problem and the probabilities of transition from one state to another.

- `executions.py`: This file contains the `Environnement` class that is used to initialize and run the environment.

Overall, these files work together to define the behavior of the environment and provide the necessary information for the agent to learn and make decisions based on the observed state. 

# Running the Project

To run the project, navigate to the `main.py` file and execute it. This file contains the main code that orchestrates the interaction between the agent and the environment. Specifically, it imports the necessary modules and objects from the `agents` and `env` directories, initializes the environment, and trains the agent on the given task.

Make sure that you have all the necessary libraries installed before running the code. You can check the list of libraries used in the project in the `README` file.

Once you have executed `main.py`, you should be able to see the agent learning and making decisions based on the observed state. The output of the program will depend on the specific implementation of the agent and the environment, but should give you an idea of the performance of the system and the behavior of the agent under different conditions.

















