# Perceptron

## LIBRARIES

import numpy as np


## MAIN FUNCTIONS

def importData(filename):
    """
    Imports dataset

    Parameters:
        filename (str) - name of the file, including extension
    Returns:
        features (array), labels (array)
    """
    # split features and class labels
    labels = []
    features = []

    with open(filename) as file:
        for line in file:
            instance = line.strip().split(',')
            
            features.append(np.array(instance[:-1], float))

            # convert class label to corresponding integer
            for char in instance[-1]:
                if(char.isdigit()):
                    labels.append(np.array(char, int))   

    return np.array(features), np.array(labels)


def assignClasses(features, labels, positiveClass, negativeClass):
    """
    Assigns one of two classes as postitive and the other as negative;
    changes class labels to +1 or -1 accordingly 

    Parameters:
        features (array)
        labels (array)
        positiveClass (int) - 1, 2 or 3
        negativeClass (int) - 1, 2 or 3 (must be different to positiveClass)
    Returns:
        featuresSubset (array), labelsBinary (array)
    """
    
    # empty arrays to be populated
    featuresSubset = np.empty((0,4), float)
    labelsBinary = np.empty((0,1), int)

    for instance in range(len(features)):
        X = features[instance]
        y = labels[instance]

        # if X is in specified class, add to array and change y accordingly
        if(y == positiveClass):
            featuresSubset = np.vstack((featuresSubset, X))
            labelsBinary = np.vstack((labelsBinary, 1))

        elif(y == negativeClass):
            featuresSubset = np.vstack((featuresSubset, X))
            labelsBinary = np.vstack((labelsBinary, -1))

    # shuffle data
    np.random.seed(1)
    permutation = np.random.permutation(len(featuresSubset))
    featuresSubset = featuresSubset[permutation]
    labelsBinary = labelsBinary[permutation]

    return featuresSubset, labelsBinary


def mcAssignClasses(features, labels, positiveClass):
    """
    Assigns specified class as postitive and all others as negative;
    changes class labels to +1 or -1 accordingly 

    Parameters:
        features (array)
        labels (array)
        positiveClass (int) - 1, 2 or 3
    Returns:
        featuresSubset (array), labelsBinary(array)
    """
    # empty arrays to be populated
    featuresSubset = np.empty((0,4), float)
    labelsBinary = np.empty((0,1), int)

    for instance in range(len(features)):
        X = features[instance]
        y = labels[instance]

        # if X is in specified class, change y to +1
        if(y == positiveClass):
            featuresSubset = np.vstack((featuresSubset, X))
            labelsBinary = np.vstack((labelsBinary, 1))
        # otherwise, y = -1
        else:
            featuresSubset = np.vstack((featuresSubset, X))
            labelsBinary = np.vstack((labelsBinary, -1))

    # shuffle data    
    np.random.seed(2)
    permutation = np.random.permutation(len(features))
    featuresSubset = featuresSubset[permutation]
    labelsBinary = labelsBinary[permutation]
        
    return featuresSubset, labelsBinary


def perceptronTrainer(trainingFeatures, trainingLabels, noIterations):
    """
    Trains the peceptron over a particular number of iterations

    Parameters:
        trainingFeatures (array)
        trainingLabels (array)
        noIterations (int) - number of iterations required
    Returns:
        b - bias
        W - weights vector
    """
    # initialise weights and bias
    W = np.zeros(4) 
    b = 0

    for iteration in range(noIterations):
        for instance in range(len(trainingFeatures)):
        
            X = trainingFeatures[instance]
            y = trainingLabels[instance]
            
            # compute activation score for each instance
            a = np.dot(W,X) + b

            #if instance is misclassified, update W and b
            if(y*a <= 0):
                W += y*X
                b += y
            
    return b, W


def mcL2Trainer(trainingFeatures, trainingLabels, noIterations, lam):
    """
    Trains the peceptron over a particular number of iterations;
    updates weights vector according to the value of lam

    Parameters:
        trainingFeatures (array)
        trainingLabels (array)
        noIterations (int) - number of iterations required
        lam (float) - regularisation coefficient (lambda)
    Returns:
        b - bias
        W - weights vector
    """
    # initialise weights and bias
    W = np.zeros(4) 
    b = 0

    for iteration in range(noIterations):
        for instance in range(len(trainingFeatures)):
        
            X = trainingFeatures[instance]
            y = trainingLabels[instance]
            
            # compute activation score for each instance
            a = np.dot(W,X) + b

            #if instance is misclassified, update W with L2 regularisation term and b as normal
            if(y*a <= 0):
                W = (1-2*lam)*W + y*X
                b += y
            
    return b, W


def predict(b, W, X):
    """
    Predicts whether an instance belongs to the postive or negative class

    Parameters:
        b - bias
        W - weights vector
        X (array) - features of one instance
    Returns:
        sign(a) - +1, 0 or -1
    """
    a = np.dot(W,X) + b
    
    return np.sign(a)


def mcPredict(b1, W1, b2, W2, b3, W3, X):
    """
    Predicts which class an instance belongs to

    Parameters:
        b1, b2, b3 - biases
        W1, W2, W3 - weights vectors
        X (array) - features of one instance
    Returns:
        y (int) - class label
    """
    scores = []

    # compute activation scores with different b,W combinations
    a1 = np.dot(W1,X) + b1
    scores.append(a1)

    a2 = np.dot(W2,X) + b2
    scores.append(a2)

    a3 = np.dot(W3,X) + b3
    scores.append(a3)
    
    # highest score determines label
    y = np.argmax(scores) + 1

    return y


def perceptronAccuracy(features, labels, b, W):
    """
    Calculates the accuracy of the perceptron

    Parameters:
        features (array)
        labels (array)
        b - bias
        W - weights vector
    Returns:
        accuracy (float) - value between 0 and 1
    """
    correctPredictions = 0

    for instance in range(len(features)):
        X = features[instance]
        y = labels[instance]

        # if signs are the same, prediction is correct
        if(np.sign(y) == predict(b, W, X)):
            correctPredictions += 1

    # calculate accuracy
    accuracy = correctPredictions/len(features)

    return accuracy


def mcAccuracy(features, labels, b1, W1, b2, W2, b3, W3):
    """
    Calculates the accuracy of the multi-class classifier

    Parameters:
        features (array)
        labels (array)
        b1, b2, b3 - biased
        W1, W2, W3 - weights vectors
    Returns:
        accuracy (float) - value between 0 and 1
    """
    correctPredictions = 0

    for instance in range(len(features)):
        X = features[instance]
        y = labels[instance]

        # if class labels are the same, prediction is correct
        if(y == mcPredict(b1, W1, b2, W2, b3, W3, X)):
            correctPredictions += 1
    
    # calculate accuracy
    accuracy = correctPredictions/len(features)

    return accuracy



## AUXILLIARY FUNCTIONS

def runPerceptron():
    # create subsets for both training and test data
    trainingFeatures1_2, trainingLabels1_2 = assignClasses(trainingFeatures, trainingLabels, 1, 2)
    trainingFeatures2_3, trainingLabels2_3 = assignClasses(trainingFeatures, trainingLabels, 2, 3)
    trainingFeatures1_3, trainingLabels1_3 = assignClasses(trainingFeatures, trainingLabels, 1, 3)

    testFeatures1_2, testLabels1_2 = assignClasses(testFeatures, testLabels, 1, 2)
    testFeatures2_3, testLabels2_3 = assignClasses(testFeatures, testLabels, 2, 3)
    testFeatures1_3, testLabels1_3 = assignClasses(testFeatures, testLabels, 1, 3)

    # print results
    print("\n**********")
    print("PERCEPTRON")
    print("**********\n")

    print(" * CLASS 1 & 2 *\n")
    b1_2, W1_2 = perceptronTrainer(trainingFeatures1_2, trainingLabels1_2, 20)
    print("Bias:", b1_2, "Weights:", W1_2)

    trainAccuracy1_2 = perceptronAccuracy(trainingFeatures1_2, trainingLabels1_2, b1_2, W1_2)
    print("Training accuracy using class 1 & 2: {:.2f}".format(trainAccuracy1_2))

    testAccuracy1_2 = perceptronAccuracy(testFeatures1_2, testLabels1_2, b1_2, W1_2)
    print("Test accuracy using class 1 & 2: {:.2f}".format(testAccuracy1_2))
    print()
    print()

    print(" * CLASS 2 & 3 *\n")
    b2_3, W2_3 = perceptronTrainer(trainingFeatures2_3, trainingLabels2_3, 20)
    print("Bias:", b2_3, "Weights:", W2_3)

    trainAccuracy2_3 = perceptronAccuracy(trainingFeatures2_3, trainingLabels2_3, b2_3, W2_3)
    print("Training accuracy using class 2 & 3: {:.2f}".format(trainAccuracy2_3))

    testAccuracy2_3 = perceptronAccuracy(testFeatures2_3, testLabels2_3, b2_3, W2_3)
    print("Test accuracy using class 2 & 3: {:.2f}".format(testAccuracy2_3))
    print()
    print()

    print(" * CLASS 1 & 3 *\n")
    b1_3, W1_3 = perceptronTrainer(trainingFeatures1_3, trainingLabels1_3, 20)
    print("Bias:", b1_3, "Weights:", W1_3)

    trainingAccuracy1_3 = perceptronAccuracy(trainingFeatures1_3, trainingLabels1_3, b1_3, W1_3)
    print("Training accuracy using class 1 & 3: {:.2f}".format(trainingAccuracy1_3))

    testAccuracy1_3 = perceptronAccuracy(testFeatures1_3, testLabels1_3, b1_3, W1_3)
    print("Test accuracy using class 1 & 3: {:.2f}".format(testAccuracy1_3))
    print()


def runMC():
    # create subsets for both training and test data
    trainingFeatures1, trainingLabels1 = mcAssignClasses(trainingFeatures, trainingLabels, 1)
    trainingFeatures2, trainingLabels2 = mcAssignClasses(trainingFeatures, trainingLabels, 2)
    trainingFeatures3, trainingLabels3 = mcAssignClasses(trainingFeatures, trainingLabels, 3)

    # print results
    print()
    print("\n*****************")
    print("MC CLASSIFICATION")
    print("*****************\n")

    b1, W1 = perceptronTrainer(trainingFeatures1, trainingLabels1, 20)
    print("Bias 1:", b1, "Weights 1:", W1)

    b2, W2 = perceptronTrainer(trainingFeatures2, trainingLabels2, 20)
    print("Bias 2:", b2, "Weights 2:", W2)

    b3, W3 = perceptronTrainer(trainingFeatures3, trainingLabels3, 20)
    print("Bias 3:", b3, "Weights 3:", W3)

    accuracy = mcAccuracy(trainingFeatures, trainingLabels, b1, W1, b2, W2, b3, W3)
    print("\nTraining accuracy of multi-class classifier: {:.2f}".format(accuracy))

    accuracy = mcAccuracy(testFeatures, testLabels, b1, W1, b2, W2, b3, W3)
    print("Test accuracy of multi-class classifier: {:.2f}".format(accuracy))
    print()


def runMCL2():
    # create subsets for both training and test data
    trainingFeatures1, trainingLabels1 = mcAssignClasses(trainingFeatures, trainingLabels, 1)
    trainingFeatures2, trainingLabels2 = mcAssignClasses(trainingFeatures, trainingLabels, 2)
    trainingFeatures3, trainingLabels3 = mcAssignClasses(trainingFeatures, trainingLabels, 3)

    # print results
    print()
    print("\n*************************")
    print("MC CLASSIFICATION with L2")
    print("*************************\n")

    print("* Lambda = 0.01 *\n")
    b1, W1 = mcL2Trainer(trainingFeatures1, trainingLabels1, 20, 0.01)
    print("Bias 1:", b1, "Weights 1:", W1)

    b2, W2 = mcL2Trainer(trainingFeatures2, trainingLabels2, 20, 0.01)
    print("Bias 2:", b2, "Weights 2:", W2)

    b3, W3 = mcL2Trainer(trainingFeatures3, trainingLabels3, 20, 0.01)
    print("Bias 3:", b3, "Weights 3:", W3)

    accuracy = mcAccuracy(trainingFeatures, trainingLabels, b1, W1, b2, W2, b3, W3)
    print("\nTraining accuracy of multi-class classifier when lambda = 0.01: {:.2f}".format(accuracy))

    accuracy = mcAccuracy(testFeatures, testLabels, b1, W1, b2, W2, b3, W3)
    print("Test accuracy of multi-class classifier when lambda = 0.01: {:.2f}".format(accuracy))
    print()
    print()

    print("* Lambda = 0.1 *\n")
    b1, W1 = mcL2Trainer(trainingFeatures1, trainingLabels1, 20, 0.1)
    print("Bias 1:", b1, "Weights 1:", W1)

    b2, W2 = mcL2Trainer(trainingFeatures2, trainingLabels2, 20, 0.1)
    print("Bias 2:", b2, "Weights 2:", W2)

    b3, W3 = mcL2Trainer(trainingFeatures3, trainingLabels3, 20, 0.1)
    print("Bias 3:", b3, "Weights 3:", W3)

    accuracy = mcAccuracy(trainingFeatures, trainingLabels, b1, W1, b2, W2, b3, W3)
    print("\nTraining accuracy of multi-class classifier when lambda = 0.1: {:.2f}".format(accuracy))

    accuracy = mcAccuracy(testFeatures, testLabels, b1, W1, b2, W2, b3, W3)
    print("Test accuracy of multi-class classifier when lambda = 0.1: {:.2f}".format(accuracy))
    print()
    print()

    print("* Lambda = 1.0 *\n")
    b1, W1 = mcL2Trainer(trainingFeatures1, trainingLabels1, 20, 1.0)
    print("Bias 1:", b1, "Weights 1:", W1)

    b2, W2 = mcL2Trainer(trainingFeatures2, trainingLabels2, 20, 1.0)
    print("Bias 2:", b2, "Weights 2:", W2)

    b3, W3 = mcL2Trainer(trainingFeatures3, trainingLabels3, 20, 1.0)
    print("Bias 3:", b3, "Weights 3:", W3)

    accuracy = mcAccuracy(trainingFeatures, trainingLabels, b1, W1, b2, W2, b3, W3)
    print("\nTraining accuracy of multi-class classifier when lambda = 1.0: {:.2f}".format(accuracy))

    accuracy = mcAccuracy(testFeatures, testLabels, b1, W1, b2, W2, b3, W3)
    print("Test accuracy of multi-class classifier when lambda = 1.0 : {:.2f}".format(accuracy))
    print()
    print()

    print("* Lambda = 10.0 *\n")
    b1, W1 = mcL2Trainer(trainingFeatures1, trainingLabels1, 20, 10.0)
    print("Bias 1:", b1, "Weights 1:", W1)

    b2, W2 = mcL2Trainer(trainingFeatures2, trainingLabels2, 20, 10.0)
    print("Bias 2:", b2, "Weights 2:", W2)

    b3, W3 = mcL2Trainer(trainingFeatures3, trainingLabels3, 20, 10.0)
    print("Bias 3:", b3, "Weights 3:", W3)

    accuracy = mcAccuracy(trainingFeatures, trainingLabels, b1, W1, b2, W2, b3, W3)
    print("\nTraining accuracy of multi-class classifier when lambda = 10.0: {:.2f}".format(accuracy))

    accuracy = mcAccuracy(testFeatures, testLabels, b1, W1, b2, W2, b3, W3)
    print("Test accuracy of multi-class classifier when lambda = 10.0: {:.2f}".format(accuracy))
    print()
    print()

    print("* Lambda = 100.0 *\n")
    b1, W1 = mcL2Trainer(trainingFeatures1, trainingLabels1, 20, 100.0)
    print("Bias 1:", b1, "Weights 1:", W1)

    b2, W2 = mcL2Trainer(trainingFeatures2, trainingLabels2, 20, 100.0)
    print("Bias 2:", b2, "Weights 2:", W2)

    b3, W3 = mcL2Trainer(trainingFeatures3, trainingLabels3, 20, 100.0)
    print("Bias 3:", b3, "Weights 3:", W3)

    accuracy = mcAccuracy(trainingFeatures, trainingLabels, b1, W1, b2, W2, b3, W3)
    print("\nTraining accuracy of multi-class classifier when lambda = 100.0: {:.2f}".format(accuracy))

    accuracy = mcAccuracy(testFeatures, testLabels, b1, W1, b2, W2, b3, W3)
    print("Test accuracy of multi-class classifier when lambda = 100.0: {:.2f}".format(accuracy))
    print()


#####################################################################

## import training and test data here
##trainingFeatures, trainingLabels = importData("train.data")
#testFeatures, testLabels = importData("test.data")

#####################################################################


#####################################################################

#runPerceptron()
#runMC()
#runMCL2()

#####################################################################

