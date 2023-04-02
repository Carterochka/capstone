from data import ClassicalMechanicsDataset
import torch
import numpy as np
import re
import os
from itertools import product
from tqdm import tqdm

import phyre

class ThreeBallsCollisionFullSceneDataset(ClassicalMechanicsDataset):
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        return torch.FloatTensor(data[0]).unsqueeze(dim=0), torch.FloatTensor(data[1:, [0, 1, 3, 4, 6, 7]]).reshape(1,-1)


    def generate_data(self, max_actions=30000, **kwargs):
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

        # Defining a function to check if the red ball is in the free fall throughout the simulation
        def is_red_ball_in_free_fall(simulation):
            features = simulation.featurized_objects.features
            return False not in [features[0][-1][0] == features[frame_id][-1][0] for frame_id in range(len(features))]

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
        
        # getting the free-fall fraction from the function call arguments
        free_fall_fraction = kwargs['free_fall_fraction']

        # setting up helper variables to manage the fraction of free-fall scenarios
        free_fall_paths = []
        total_count = 0

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        else:
            import glob
            files = glob.glob(self.data_path + '/*')
            for file in files:
                os.remove(file)

        for task_index, action_index in tqdm(product(range(len(tasks)), range(len(actions))), desc='Data generation progress: '):
            simulation = simulator.simulate_action(task_index, actions[action_index], need_images=True, need_featurized_objects=True, stride=15)
            if simulation.status.is_invalid(): continue
            simulation_data = get_simulation_data(simulation)
            if not simulation_data: continue
            file_path = self.data_path + f'/task-{task_index}-action-{action_index}'

            # If the simulated scenario is the free-fall, then save its path
            if is_red_ball_in_free_fall(simulation):
                free_fall_paths.append(file_path)
            
            # Save the simulation and count it toward the total
            np.save(file_path, simulation_data)
            total_count += 1

        # Getting the number of scenarios that need to be deleted
        delete_num = int((len(free_fall_paths) - free_fall_fraction*total_count) / (1 + free_fall_fraction))

        # Deleting the number of free-fall scenarios calculated above
        delete_paths = np.random.choice(free_fall_paths, delete_num)
        map(os.remove, delete_paths)
        map(free_fall_paths.remove, delete_paths)
        total_count -= delete_num

        print(f'Total scenarios generated: {total_count}')
        print(f'Fraction of the free-fall scenarios: {free_fall_fraction}')