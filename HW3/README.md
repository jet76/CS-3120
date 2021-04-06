#Assignment 3 - Class Animals classification using K-Nearest Neighbor classifier   
(NumPy and data visualization packages are allowed.)  
(SKLearn models are allowed.)  
**
Given:  
K = [3, 5, 7]  
Distance metrics: L1 and L2  
Dataset: animals.zip
**

1.	**Gather the Dataset:** The Animals datasets consists of 3,000 images with 1,000 images per dog, cat, and panda class, respectively. Each image is represented in the RGB color space. You will preprocess each image by resizing it to 32x32 pixels. Taking into account the three RGB channels, the resized image dimensions imply that each image in the dataset is represented by 32x32x3 = 3,072 integers. (2pts code)

2.	**Split the Dataset**: You’ll be using three splits of the data. One split for training, one split for validation and the other for testing. Please randomly partition the data into these three splits. For example, 70% for training, 10% for validation and 20% for testing. Report your final performance only using the testing dataset.

3.	Your k-NN classifier will be trained on the raw pixel intensities of the images in the training set. You need to convert the images to data vectors with label.  

4.	**Train the Classifier**: k-NN classifier from sklearn or your own function could be used to train the model. (2pts code) 

5.	**Evaluate**: Once your k-NN classifier is trained, you should evaluate performance (accuracy, precision, recall, F-measure) on the test set. These scores need to be included in your report. (2pts)

6.	What is the best value of K to use? What is the best distance to use? Answer or analysis of this question needs to be included in your report. (2pts)
 
7.	**Bonus up to 2 pts** for who used own developed KNN model.
