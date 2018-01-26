# NAS
This repository is complementary to my MSc thesis project "DESIGN AND EVALUATION OF NEURAL ARCHITECTURE, USING REINFORCEMENT LEARNING AND DISTRIBUTED COMPUTING", submitted for the partial fullfilment of requirements for the MSc degree in "Computational Methods and Applications", University of Macedonia, Greece.
# Structure
The repository is organized in three sections:
- DDQN, a Keras implementation of Double Deep Q-Learning for simple, fully connected neural networks for the MNIST dataset
- A2C-DISCRETE, a Tensorflow implementation of Synchronous Advantage Actor-Critic (A2C) in discrete action spaces, for the same problem as DDQN
- A2C-CONTINUOUS, a tensorflow implementation of A2C for fully convolutional networks for the CIFAR10 dataset
# Experimental setup
The main libraries required to run the experiments are mpi4py, tensorflow, keras. A full conda environment export is available at conda.env
For each project file, in the tests folder a complete copy of the script ran (A2C.py or DDQN.py) is saved, as well as a .csv with the results of each episode.
In order to run the experiments, A2C.py or DDQN.py must be executed. On the continuous action space project, there are two available cli options: the first concerns the dataset used and is positional (mandatory). The available options are mnist or cifar10 the second option is optional (-t) and denotes that the generated architectures will be fully trained for evaluation.
