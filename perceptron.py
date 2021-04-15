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


