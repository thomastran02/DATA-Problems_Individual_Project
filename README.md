# DATA-Problems_Individual_Project

![image](https://user-images.githubusercontent.com/98187543/226465520-2d0820b4-ed17-44d7-afaa-5fccf149c778.png)

# Spotify Million Playlist Dataset

* **One Sentence Summary** This repository holds an attempt to build a recommendation system for Spotify users based on the Spotify Million Playlist Dataset. (https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#task). 

## Overview

  * **Definition of the tasks / challenge:** The task is to build a recommendation system for Spotify Playlist. Each playlist in the training set contains between 5-100 tracks.
  * **Approach:** This repository contains two approaches toward the problem, the first one utilized creating a matrix with implicit ratings of each track in the playlist for each user. Then, using a cosine similarity score to determine which track to recommend to the user. The second approach utilizes matrix factorization to try and predict how users would rate different tracks and recommend tracks the highest rated tracks the user hasn't listened to yet.
  
  * **Summary of the performance achieved** Compare the two different methods on a test set that takes out a certain amount of tracks and sees how accurately the recommendation system works. Testing different hyperparameters in the cosine similarity approach as well as run-time between different methods

## Summary of Workdone

### Data

* Data:
  * Type:
    * Input: Row containing ratings from a user for each track.
  * Size: The total size of the challenge dataset is 10,000 playlists. 
  * Instances: Use 900 playlist for training the recommendation system and use 100 playlist for testing.

#### Preprocessing / Clean up

* Worked on a subset of the full dataset (about 10% of the full dataset) to reduce computation time.
* Reformatted the data by creating a matrix where each row is a user and each column contains an implicit rating for each track. The implicit rating was found by how many times the artist appears in relation to the size of the track. Therefore, ratings for each track must be between 0 and 1.
* The preprocessing / Clean up portion of this task accounts for the bulk of the work, one reason is because of the size of the original dataset had to be reduced to fit my system. Another reason is because it required having to create a rating matrix as it is not explicitly given.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Method 1: K-Nearest Neighbor Algorithm Cosine Similarity
  * Input: User ratings for each track
  * Output: Predict ratings for tracks user hasn't rated
  * Models
    * Using cosine similarity, can find the similarity scores between users. The cosine similarity score is given by:
![image](https://user-images.githubusercontent.com/98187543/228328679-baa2cabe-a8e6-4e2f-88e8-dc69f127ecb3.png)
    * After finding the similarity scores for each user, can use K-nearest neighbor algorithm to determine which track to recommend.
    * Hyperparameter: k-neighbors, this repository utilized k=5,k=50 and k=3

* Method 2: Matrix Factorizaion
  * Input: Rating matrix
  * Output: 2 Matrices who's dot product attempts to mimic user behaviour by predicting features of users and items
  * Models
    * Assuming 10 features attemps to factor the matrix into n*10 and 10*m matrix:
 ![image](https://user-images.githubusercontent.com/98187543/235410195-171efae9-e2d4-4b44-b4c7-0052f4c61c12.png)
   * Hyperparameter: The number of iterations is 100 with a learning rate of 0.01 and regularation parameter is 0.01. This algorithm has early stopping when the error is 0.01


### Training

* Method 1:
  * This involved running the rating matrix and similarity matrix through an algorithm that calculates the highest tracks based on the most similar users
  * The algorithm involved in this method did not require training a model, it just involved creating the similarity matrix and then testing the algorithm with different k-values
  * This repository only attempts 3 different values of k, k=3 appears to contain the best result using this method.
* Method 2:
  * This method involved an algorithm that attempted to factor the rating matrix into 2 smaller matrices which attempt to represent 10 unknown features for the users and items.
  * Training involved using a maximum of 5000 iterations, a learning rate of 0.001 and a regularation parameter of 0.01. The algorithm stopped early if the error reached 0.001.
  * At first tried to involve using 5000 iterations, a learning rate of 0.001 and a regularation parameter of 0.01 with an early stopping 0.001. However, using these parameter was taking a very long time to train and so the parameters were changed to 100 iterations, learning rate 0.01, regularation parameter 0.01 and early stopping 0.01 to reduce training time.

* One challenge with this dataset was the size of the original dataset, my current system does not have the capacity to run the algorithm that I created on the whole dataset and therefore required me to take a subset of the data. Even using a subset of the original dataset the training still took a long time.

### Performance Comparison

* The metric used to compare the results is precision:

![image](https://user-images.githubusercontent.com/98187543/235411947-7eb0696c-d19e-4a02-bc3f-38185bb3eec8.png)

  * 100 playlist that contained 100 tracks were chosen randomly from the original dataset and 50 randomly selected tracks were removed from these playlist.
  * Then utilizing the top 5 recommended tracks a precision score was found based on the number of tracks recommended that were in the original playlist.
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* Training for method 2 can definitely be reduced by utilizing the tensor flow library as opposed to just numpy library. Therefore, the next step would involve optimizing the ideas utilized in this repository to decrease the training time significantly.

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
