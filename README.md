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
  * Output: 2 Matrices who's dot product mimics user behaviour by predicting features of users and items
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
  * At first tried to involve using 5000 iterations, a learning rate of 0.001 and a regularation parameter of 0.01 with an early stopping 0.001. However, using these parameter was taking days to train and so the parameters were changed to 100 iterations, learning rate 0.01, regularation parameter 0.01 and early stopping 0.01 to reduce training time.

* One challenge with this dataset was the size of the original dataset, my current system does not have the capacity to run the algorithm that I created on the whole dataset and therefore required me to take a subset of the data. Even using a subset of the original dataset the training still took a long time.

### Performance Comparison

* The metric used to compare the results is precision:

![image](https://user-images.githubusercontent.com/98187543/235411947-7eb0696c-d19e-4a02-bc3f-38185bb3eec8.png)

  * 100 playlist that contained 100 tracks were chosen randomly from the original dataset and 50 randomly selected tracks were removed from these playlist.
  * Then utilizing the top 5 recommended tracks a precision score was found based on the number of tracks recommended that were in the original playlist.
![image](https://user-images.githubusercontent.com/98187543/235493315-3a972532-b104-41eb-865e-a0f0b05e76bb.png)

### Conclusions

* Based on our results the cosine similarity method with k=3 produced the best precision with 0.15 and a run time of 1.31 seconds. However, I think method 2 can be better optimized and tune the hyperparameters to fit the problem better. The adjustments I made in this attempt were due to a lack of computation power and modified to reduce run time.

### Future Work

* Training for method 2 can be reduced by utilizing the tensor flow library as opposed to just numpy library. Therefore, the next step would involve optimizing the ideas utilized in this repository to decrease the training time significantly as well as optimizing the hyperparameters for method 2.
* Furthermore, the more in depth challenge utilizes the names of the playlist and sentiment analysis to attempt to recommend songs based on the correlation between songs.

## How to reproduce results

* The main challenge with this particular problem involved the preprocessing stage and creating a rating matrix based on how the users interacted with the items.
* This attempt utilized an implicit rating, by counting the number of times a particular artist appeared in a playlist and associating it with their song to give a rating, then normalizing that result. For example, if one artist's songs appear 2 times in a playlist that contains 5 songs, each of the 2 songs will receive a rating of 0.4. This implicit rating was then used to create the rating matrix.
* For method 1, utilize some similarity metric to create a similarity score between users. This repository utilized the cosine similarity.
* For method 2, develop an algorithm possibly using tensor flow or some deep learning model to attempt to factor the rating matrix into 2 matrices that contain hidden features for the users and the items.

### Overview of files in repository

* List of files:
  * Cosine_Similarity.ipynb: contains the algorithms and code that was utilized in method 1 in this repository
  * Matrix_Factorization.ipynb: contains algorithms and code that was utilized in method 2 in this repository
  * Data_Visualization.ipynb: Some tables and visualizations of the correlation between certain artist within the playlists.

### Software Setup
* Standard libraries used

### Data
* The data can be found at: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#task

### Training

* No training necessarily involved in method 1
* Future attempts of method 2 should attempt to utilize the tensor flow library to reduce run time, tune hyperparameters and better optimize the results.

## Citations

https://www.geeksforgeeks.org/pandas-parsing-json-dataset/

https://www.youtube.com/watch?v=xySjbVUgAwU

C.W. Chen, P. Lamere, M. Schedl, and H. Zamani. Recsys Challenge 2018: Automatic Music Playlist Continuation. In Proceedings of the 12th ACM Conference on Recommender Systems (RecSys ’18), 2018.

https://developers.google.com/machine-learning/recommendation/collaborative/basics

https://www.youtube.com/watch?v=ZspR5PZemcs

https://developers.google.com/machine-learning/recommendation/collaborative/basics

https://pynative.com/python-save-dictionary-to-file/

https://towardsdatascience.com/collaborative-filtering-based-recommendation-systems-exemplified-ecbffe1c20b1

https://www.youtube.com/watch?v=cxcFi3RDrEw

https://towardsdatascience.com/recommendation-systems-explained-a42fc60591ed

https://www.youtube.com/watch?v=DYj4T4SZAQc
