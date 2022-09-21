'''
    This script is generating a dataset of a single ball in free fall.
    For this purpose, we choose a single scene from the simulator and
    populating 10,000 random valid actions. We only keep those that
    result in a valid simulation without any collisions with other objects.

    This script should be run from project root directory.
'''

import numpy as np
import math
import re
import itertools
import os, sys

import phyre

# Choosing a setup where only one ball is needed
eval_setup = 'ball_cross_template'


# We only need one fold as we mix all the data together anyway
fold_id = 0

train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
tasks = list(train_tasks + dev_tasks + test_tasks)

# Printing output for logging purposes
print('Total tasks: ', len(tasks))

# Filtering tasks to only include a simple two-ball template. The template key: '00000:xxx'

task_filter = re.compile("00000:*")
tasks = list(filter(task_filter.match, tasks))

print('Tasks after filtering: ', len(tasks))

# Choosing a single scenario in which we will generate our free-falls
tasks = [tasks[0]]

# Getting action tier for our tasks - a single ball
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
print('Action tier for', eval_setup, 'is', action_tier)

# Create the simulator from the tasks and tier.
simulator = phyre.initialize_simulator(tasks, action_tier)

# getting a 10000 actions from a simulator
# it uniformly samples actions skipping invalid ones
# Action dimensions: 3 (x, y, radius) - represent coordinates and size of the red ball

actions = simulator.build_discrete_action_space(max_actions=10000)

# Defining a function to check if the red ball is in the free fall throughout the simulation
def is_red_ball_in_free_fall(simulation):
    features = simulation.featurized_objects.features
    return False not in [features[0][-1][0] == features[frame_id][-1][0] for frame_id in range(len(features))]

# Getting only the coordinates of the red ball
def get_red_ball_data(simulation):
    features = simulation.featurized_objects.features
    data = []
    for frame_id in range(len(features)):
        data.append([features[frame_id][-1][0], features[frame_id][-1][1]])
    return data

# we are using a single task
task_index = 0

# Creating the directory for the dataset if it doesn't yet exist
path = 'data/single-ball-free-fall'
if not os.path.exists(path):
    os.makedirs(path)
else:
    import glob
    files = glob.glob(path + '/*')
    for file in files:
        os.remove(file)

for action_index in range(len(actions)):
    simulation = simulator.simulate_action(task_index, actions[action_index], need_images=True, need_featurized_objects=True, stride=15)
    if simulation.status.is_invalid(): continue
    if is_red_ball_in_free_fall(simulation):
        np.save(f'data/single-ball-free-fall/action-{action_index}', get_red_ball_data(simulation))