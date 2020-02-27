import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix


def fit_perceptron(X_train, y_train):
    max_iter = 1000  # max amount of times we will run the pocket algorithm
    x = np.array(X_train)  # convert X_train into a numpy array
    a = np.ones((x.shape[0], 1))
    x = np.hstack((a, x))  # initialize and stack x0 to the left of all x values
    w = np.zeros((x.shape[1]))  # initialize the w to d + 1 (w[0] is w0)
    w_min = [1., w]  # initialize the variable that holds the min error value and the corresponding w

    for j in range(max_iter):
        for i in range(len(x)):
            if np.sign(pred(x[i], w)) != y_train[i]:  # check if the label value is correct
                w = w + y_train[i] * np.transpose(x[i])  # if not, adjust the w
                error_val = errorPer(x, y_train, w)
                if error_val < w_min[0]:  # each time w is adjusted, check if the w value is the best for the set
                    w_min[0] = error_val  # if yes, pocket this value into w_min
                    w_min[1] = w
    return w_min[1]  # return the min pocketed w


def errorPer(X_train, y_train, w):
    errors = count = 0  # intialize the counts
    for i in range(len(X_train)):
        count += 1
        if np.sign(pred(X_train[i], w)) != y_train[i]:  # if the sign is incorrect, increment the error count
            errors += 1
    return errors / count  # return average error out of 1


def confMatrix(X_train, y_train, w):
    x = np.array(X_train)  # convert X_train into a numpy array
    a = np.ones((x.shape[0], 1))
    x = np.hstack((a, x))  # initialize and stack x0 to the left of all x values
    result = np.zeros((2, 2))
    for i in range(len(x)):
        val = np.sign(pred(x[i], w))
        if y_train[i] == -1:
            if y_train[i] == val:  # value is correct and -1
                result[0][0] += 1
            else:
                result[0][1] += 1  # value is incorrect and -1
        else:
            if y_train[i] == val:
                result[1][1] += 1  # value is correct and 1
            else:
                result[1][0] += 1  # value is incorrect and 1
    return result


def pred(X_train, w):
    tx = np.transpose([X_train])  # transpose the x and dot product to find the class label
    return np.dot(w, tx)


def test_SciKit(X_train, X_test, Y_train, Y_test):
    pct = Perceptron()  # initialize perceptron
    pct.fit(X_train, Y_train)  # train it using the training data
    pred_pct = pct.predict(X_test)  # predict against the test data
    return confusion_matrix(Y_test, pred_pct)  # return the confusion matrix


def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size=0.2)

    for i in range(80):
        if y_train[i] == 1:
            y_train[i] = -1
        else:
            y_train[i] = 1
    for j in range(20):
        if y_test[j] == 1:
            y_test[j] = -1
        else:
            y_test[j] = 1

    # Testing Part 1a
    w = fit_perceptron(X_train, y_train)
    cM = confMatrix(X_test, y_test, w)

    # Testing Part 1b
    sciKit = test_SciKit(X_train, X_test, y_train, y_test)

    print("Confusion Matrix is from Part 1a is: ", cM)
    print("Confusion Matrix from Part 1b is:", sciKit)


test_Part1()

# the accuracy of my confusion matrix vs. the scikit's is very similar and often identical. Oddly enough,
# the scikit's algorithm seems to sometimes misclassify more of the test values than the one my code creates, so maybe
# it runs for fewer iterations.
# Kirill Kresling - 214687735
