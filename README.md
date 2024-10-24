# Matrix Methods for Data Analysis and Machine Learning

The following notebooks are projects in relation to a course in matrix methods
for data analysis and machine learning. The projects are all done in Julia.

## Project 1

### Exercise 1

The aim is to implement different Partial Least Squares Regression (PLSR)
algorithms from
[this paper](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.2898)
and compare the performance by means of leave-one-out cross validation (LooCV)
when applied on Near Infrared (NIR) spectroscopy data.

### Exercise 2

The aim is to do a simplified analysis described in
[this paper](https://www.sciencedirect.com/science/article/pii/S003991401930709X?via%3Dihub).
We do Principal Component Analysis (PCA) to explore the data, and create
"global" and "local" PLSR models. The model performance is assessed by LooCV by
looking at RMS prediction errors. Furthermore, we compare the PLSR model with a
Ridge regression model and suggest a Multiple Objective Least Squares (MOLS)
strategy to make "local" Ridge regression models.

### Exercise 3

The aim is to suggest a PCR, PLS and Ridge regression model for multi-response
data. We also suggest a Ridge-/Tikhonov regression model for faster LooCV
calculation and compare it to the former suggested models.

## Project 2

Project 2 is split into three parts:

1. Clustering and classification of MNIST
2. Tensor compression
3. Prinicpal Variable Selection

### Exercise 1:
1. **PCA by SVD for Digit-Groups**:  
   - Perform PCA via Singular Value Decomposition (SVD) for each digit group and monitor the explained variance as a function of the number of principal components. Determine how many components are needed to explain 90% of the variance.
   - Visualize the first 10 principal component loadings as images and provide commentary.
   - Compare original images with their compressed versions after projection onto the subspace explaining 90% of the variance.

### Exercise 2:
1. **Principal Component Regression for Multiple Responses**:  
   - Extend a PCR function to handle multiple response variables and calculate the regression coefficients for each response.
   - Use this extended PCR to create a regression-based classifier with 30 principal components. Evaluate classification accuracy and confusion matrices for both the training and test sets.

### Exercise 3:
1. **K-Means Clustering within Digit-Groups**:  
   - For each digit group, apply K-means clustering with k = 3. Visualize the cluster centers and inspect PCA-score plots based on Exercise 1 results, with samples colored by cluster labels.

### Exercise 4:
1. **Repeat K-Means with Alternative Features**:  
   - Perform K-means clustering using original pixel features and random matrix-derived features. Compare results using different feature sets, including those from Non-negative Matrix Factorization (optional).
   - Identify misclassified samples, extend the feature set, and analyze improvements in test set classification.
   - Propose and test a strategy for solving a 10-class classification problem using clustering results from a 30-class classification model, both with original and extended features.

### Exercise 5:
1. **Exploratory EEF Plotting**:  
   - Use the `matread()` function to read data and reshape it into a tensor for analysis. Plot excitation-emission fluorescence (EEF) landscapes for selected samples, aiming to identify distinctive patterns using surface plots.

### Exercise 6:
1. **Tensor Decomposition and Visualization**:  
   - Perform Higher-Order Singular Value Decomposition (HOSVD) on the dataset and visualize the results. This includes plotting 2D and 3D score plots, spectra of loadings, and surface plots of the EEF landscapes. Compare the results to the raw data and adjust signs if needed.

### Exercise 7:
1. **PCR with Cross-Validation**:  
   - Conduct Principal Component Regression with 10-fold cross-validation to predict the Color response from the dataset. Plot RMSEP values and analyze the difference between using matrix X and components from HOSVD of tensor A.

### Exercise 8:
1. **Mayonnaise Spectra Analysis**:  
   - Read Near-Infrared (NIR) spectroscopy data from a mayonnaise dataset and visualize the spectra of training samples.

### Exercise 9:
1. **Principal Variable Selection & Classification**:  
   - Perform Principal Variable Selection on NIR data to select up to 10 variables. Use Linear Discriminant Analysis (LDA) to classify oil types and plot the proportion of misclassified samples in the test set. Convert the response data as needed for LDA analysis.
