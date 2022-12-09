from data import ClassicalMechanicsDataset
import torch
import numpy as np
import re
import os
from itertools import product
from tqdm import tqdm

import phyre

class ThreeBallsCollisionsDataset(ClassicalMechanicsDataset):
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        return torch.FloatTensor(data[0]).unsqueeze(dim=0), torch.FloatTensor(data[1:, -3:-1]).reshape(1,-1)


    def generate_data(self, max_actions=30000):
        # Choosing a setup where only one ball is needed
        eval_setup = 'ball_cross_template'

        # We only need one fold as we mix all the data together anyway
        fold_id = 0

        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
        tasks = list(train_tasks + dev_tasks + test_tasks)
        
        # Filtering tasks to only include a simple two-ball template. The template key: '00000:xxx'
        task_filter = re.compile("00000:*")
        tasks = list(filter(task_filter.match, tasks))

        # Getting action tier for our tasks - a single ball
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)

        # Create the simulator from the tasks and tier.
        simulator = phyre.initialize_simulator(tasks, action_tier)

        # getting a 10000 actions from a simulator
        # it uniformly samples actions skipping invalid ones
        # Action dimensions: 3 (x, y, radius) - represent coordinates and size of the red ball
        actions = simulator.build_discrete_action_space(max_actions=max_actions)

        # Getting only the coordinates of the red ball
        def get_simulation_data(simulation):
            features = simulation.featurized_objects.features
            if len(features) < 25:
                return False

            data = []
            # I am only interested in saving the first 25 frames if the simulation is larger
            for frame_id in range(25):
                instance = []
                for ball_id in range(len(features[frame_id])):
                    instance += [features[frame_id][ball_id][0], features[frame_id][ball_id][1], features[frame_id][ball_id][3]]
                data.append(instance)
            return data
        
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        else:
            import glob
            files = glob.glob(self.data_path + '/*')
            for file in files:
                os.remove(file)

        for task_index, action_index in tqdm(product(range(len(tasks)), range(len(actions)))):
            simulation = simulator.simulate_action(task_index, actions[action_index], need_images=True, need_featurized_objects=True, stride=15)
            if simulation.status.is_invalid(): continue
            simulation_data = get_simulation_data(simulation)
            if not simulation_data: continue
            np.save(self.data_path + f'/task-{task_index}-action-{action_index}', simulation_data)