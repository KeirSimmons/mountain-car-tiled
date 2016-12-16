# mountaincar

AIST ML Team

Provided code based on RL presentation (formulation of algorithms leading up to a true online TD(lambda) method using coarse coding as a function approximator to the OpenAI MountainCar environment).

You'll need to install the following modules:

gym (OpenAI)
Tiles (http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html)

To run the program, you need to pass it at least two parameters.

1: Number of episodes to train
2: Number of episodes to display after training (without any more training)

You can add optional parameters to show the surface plot of -q (negative reward) for the 2D state space at episode i, indexed at 0. 

For example:

`python final.py 1 1` will train on 1 episode, and then display the interface for 1 episode (on trained data)

`python final.py 1000 1 0` will train on 1000 episodes, then display the interface for 1 episode (on trained data) and show the surface plot after the 1st episode (episode 1 = index 0)

`python final.py 1000 10 0 9 99 999` will train on 1000 episodes, then display the interface for 10 subsequent episodes (on trained data) and show the surface plot after the 1st, 10th, 100th and final trained episode.

Should work out of the box once modules are installed.
