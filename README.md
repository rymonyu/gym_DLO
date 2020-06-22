# gym_DLO
An OpenAI gym environment that is used with PyBullet for learning purposes on the manipulation tasks for deformable linear object.

## Installation of this environment
Go to the gym-DLO folder and run the command -

```bash
pip install -e .
```

This will install the gym environment and all required packages.

## Usage of the gym
```bash
import gym
import gym_DLO
env = gym.make('DLO-v0')
```

## Random Perturbation of the DLO
Here is an example of the random perturbation task on a simulated rope.
<p align="center">
  <img src=https://github.com/rymonyu/gym_DLO/blob/master/Animations/perturb.gif />
<\p>

