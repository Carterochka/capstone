# Welcome to complimentary repository for my capstone project!
## Comparative Analysis of Traditional Recurrent Neural Networks and Reservoir Computing Models for Predicting Complex Ball Trajectories in a 2D Environment

The structure of this repository is the following:
- `data` module contains the scripts for data generation. If, for some reasons, you are unable to run them, you can use [this link](https://drive.google.com/file/d/1BIIKT_QceaaomOpdobUt3P6DhqS7CX71/view?usp=share_link) to download the raw data from my Google Drive (about 700 MB). Just place the "raw" folder inside the "data" folder.
- `deliverables` contains my write-ups, including the paper I submitted for my capstone.
- `models` contains importable implementations of all the deep learning models that I was using in my research.
    - Inside, there is a folder called "trained_models". It contains the trained checkpoints of traditional RNN models for the scenarios with collisions. I decided to save them, as it took more than 3 hours to train each on my laptop. Unfortunately, Reservoirpy framework doesn't provide a functionality to save ESN models, so you will need to retrain them from scratch if you will want to play with them. It took around 20-30 min to train each of the ESN models on my laptop.
- `notebooks` contains all the notebooks that document my research process. 
    - Inside, there is a folder called "playground", which I used for initial experiments. Right now this folder is mostly used for garbage collection, with most notebooks needing slight modification in the relative file paths to be done to resurrect the notebooks inside. I decided to leave the experimental notebooks in this folder instead of deleting them as they were an important part of my neural networks exploration. Let them stay at least for the memory :) 
    - There is also "other" folder, which contains some of my assignments that served as a starting point for my research project. Feel free to take a look at them if you want, but this folder also doesn't add any more value to the research findings.
- `visuals` contains some drawings of initial machine learning models that I was playing with. This folder will be removed soon!

(c) Nikita Koloskov, April 2023.
Capstone Project for Minerva University
