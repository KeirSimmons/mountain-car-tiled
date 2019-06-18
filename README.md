# MountainCar solution (OpenAI Gym)

NOTE: This was the first Python code I had ever written (when I was venturing into the Machine Learning space for the first time), hence the poor styling, code, and modularity. I am only leaving this here for nostalgic purposes and to help anyone else who is attempting this problem. If you are a recruiter looking for examples of my coding or machine learning ability, please check my other repositories, such as [this one](https://github.com/KeirSimmons/RL-Optimal-Peak-Shift).

The provided code in `final.py` is a fully working formulation of a true online TD(λ) method using coarse coding as a function approximator to the OpenAI MountainCar environment - i.e. neural networks were not used. 

## Installation Instructions

The following modules need to be installed:

* [OpenAI Gym](https://github.com/openai/gym)
* [Tiles](http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html)

## Execution Instructions

To run the program, you need to pass it at least two parameters.

1: Number of episodes to train

2: Number of episodes to display after training (without any more training)


You can add optional parameters to show the surface plot of -q (negative reward) for the 2D state space at episode i, indexed at 0. 

For example, to train on 1 episode and then display the interface for 1 episode (on trained data), use:

~~~~
python final.py 1 1
~~~~

~~~~
python final.py 1000 1 0
~~~~

will train on 1000 episodes, then display the interface for 1 episode (on trained data) and show the surface plot after the 1st episode (episode 1 = index 0).

Finally:

~~~~
python final.py 1000 10 0 9 99 999
~~~~

will train on 1000 episodes, then display the interface for 10 subsequent episodes (on trained data) and show the surface plot after the 1st, 10th, 100th and final trained episode.


Should work out of the box once modules are installed.
