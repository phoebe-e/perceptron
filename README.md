# Perceptron

## Description

Python program containing 3 types of classifier:

- Perceptron
- Multi-class classifier (1-vs-rest extension of perceptron)
- Multi-class classifier with L2 regularisation


## Operation

First of all, training and test datasets must be loaded. This can be done using the importData(filename) function.
The ‘filename’ argument should be a string (in " ") containing the name of the file to be imported (including any extension).

e.g. trainingFeatures, trainingLabels = importData(“train.data”)
     testFeatures, testLabels = importData(“test.data”)


### Perceptron

1.	After import, subsets of the data need to be generated so that the perceptron only operates on data from 2 classes at a time. The assignClasses(features, labels, positiveClass, negativeClass)
	function can be used for this. This needs to be repeated for each class combination and for both the training and test data (i.e. there should be 6 statements)

e.g. trainingFeatures1_2, trainingLabels1_2 = assignClasses(trainingFeatures, trainingLabels, 1, 2)


2.	In order to train the perceptron, the perceptronTrainer(trainingFeatures, trainingLabels, noIterations) function should be called. The bias and weights will be returned.
	This should be repeated for each class combination.

e.g. b1_2, W1_2 = perceptronTrainer(trainingFeatures1_2, trainingLabels1_2, 20)


3.	The perceptron is now ready to test. The predict(b, W, X) function is called within the perceptronAccuracy(features, labels) function and the accuracy of the predictions is returned.
	This should be repeated for each class combination.

e.g. testAccuracy1_2 = perceptronAccuracy(testFeatures1_2, testLabels1_2, b1_2, W1_2)



### Multi-Class Classifier

1.	After import, class labels need to be reassigned as 1 or -1. Three versions should be created in order to give each class a turn as the positive class. The mcAssignClasses(features, labels, positiveClass)
	function can be used for this. 

e.g. trainingFeatures1, trainingLabels1 = mcAssignClasses(trainingFeatures, trainingLabels, 1)


2.	Three sets of biases and weights should then be generated from the relabelled datasets, using the perceptronTrainer function.

e.g. b1, W1 = perceptronTrainer(trainingFeatures1, trainingLabels1, 20)


3.	The classifier can now be tested using the mcAccuracy(features, labels) function, which calls the mcPredict(b1, W1, b2, W2, b3, W3, X) function within it.

e.g. accuracy = mcAccuracy(testFeatures, testLabels, b1, W1, b2, W2, b3, W3)



### Multi-Class Classifier with L2 regularisation

1.	The three sets of training features and training labels generated in step 1 of the standard multi-class classifier can be used again here.


2.	Three sets of biases and weights need to be generated using the mcL2Trainer(trainingFeatures, trainingLabels, noIterations, lam) function, where lam is the regularisation coefficient.

e.g. b1, W1 = mcL2Trainer(trainingFeatures1, trainingLabels1, 20, 0.01)


3.	The classifier can now be tested and accuracy computed using the mcAccuracy function as before.


4.	Steps 2 and 3 can be repeated for as many values of lam as the user wishes to try.




