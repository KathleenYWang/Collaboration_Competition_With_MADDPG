[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Collaboration and Competition with MADDPG

### Introduction

For this project, we will use Multi-agent Deep Deterministic Policy Gradients to train two agents to play tennis against each other using the Unity Engine.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


### The Setup

#### 1. Setup Python environment:
Please follow the instructions in the DRLND GitHub repository [click here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### 2. Setup Unity Environment
For this project, you will not need to install Unity - this is because it has been built for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:  

1. Download the environment from one of the links below.  

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in the cloned repo and in a newly created folder named 'env', and unzip (or decompress) the file. 


### Contents

To train an agent, start with `Report.ipynb`, which contains a walkthrough of the key concepts. `maddpg.py` contains the multi-agent framework whereas `ddpg_agent.py` contains the core algorithm used to build individual DDPG agent. `networkforall.py` contains the deep neural nets that is used to model actor and critic for all. For more complex experiment, seperate networks can be used for different purposes, but for this project, only one type is used for simplicity. `checkpoint_actor.pth` and `checkpoint_critic.pth` contains the weights of the trained models. Training takes quite a long time on CPU.
