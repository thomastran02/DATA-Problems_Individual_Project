# DATA-Problems_Individual_Project

![image](https://user-images.githubusercontent.com/98187543/226465520-2d0820b4-ed17-44d7-afaa-5fccf149c778.png)

# Spotify Million Playlist Dataset

* **One Sentence Summary** This repository holds an attempt to build a recommendation system for Spotify users based on the Spotify Million Playlist Dataset. (https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#task). 

## Overview

  * **Definition of the tasks / challenge:** The task is to build a recommendation system for Spotify Playlist. Each playlist in the training set contains between 5-100 tracks.
  * **Approach:** This repository contains two approaches toward the problem, the first one utilized creating a matrix with implicit ratings of each track in the playlist for each user. Then, using a cosine similarity score to determine which track to recommend to the user. The second approach uses probability of recommending a track/artist given another track/artist is in the playlist as well.
  
  * **Summary of the performance achieved** Compare the two different methods on a test set that takes out a certain amount of tracks and sees how accurately the recommendation system works.

## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Type:
    * Input: a row containing ratings from a user for each track.
  * Size: The total size of the challenge dataset is 10,000 playlists. 
  * Instances: Use 900 playlist for training the recommendation system and use 100 playlist for testing.

#### Preprocessing / Clean up

* Worked on a subset of the full dataset (about 10% of the full dataset) to reduce computation time.
* Reformatted the data by creating a matrix where each row is a user and each column contains an implicit rating for each track. The implicit rating was found by how many times the artist appears in relation to the size of the track.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Method 1: (Use K-Nearest Neighbor)
  * Input: User ratings for each track
  * Output: Predict ratings for tracks user hasn't rated
  * Models
    * Using cosine similarity, can find the similarity scores between users. The cosine similarity score is given by:
![image](https://user-images.githubusercontent.com/98187543/228328679-baa2cabe-a8e6-4e2f-88e8-dc69f127ecb3.png)
    * After finding the similarity scores for each user, can use K-nearest neighbor algorithm to determine which track to recommend.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.
