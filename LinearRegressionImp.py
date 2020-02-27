import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    x = np.array(X_train)  # convert X_train into a numpy array
    a = np.ones((x.shape[0], 1))
    x = np.hstack((a, x))  # initialize and stack x0 to the left of all x values
    # w is the function for calculating linear regression
    w = np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.dot(np.transpose(x), np.transpose([y_train])))
    return w


def mse(X_train, y_train, w):
    x = np.array(X_train)
    if X_train[0][0] != 1:  # add the bias values for the testing case
        a = np.ones((x.shape[0], 1))
        x = np.hstack((a, x))
    count = total = 0
    for i in range(len(x)):
        val = pred(x[i], w)
        count += (val - y_train[i]) ** 2  # squaring the distance from predicted to actual y
        total += 1
    return count / total  # returning the average


def pred(X_train, w):
    w = np.transpose(w)  # transposing the arrays for the correct dot product
    xt = np.transpose([X_train])
    return np.dot(w, xt)


def test_SciKit(X_train, X_test, Y_train, Y_test):
    lin = linear_model.LinearRegression()  # initialize the linear regression model
    lin.fit(X_train, Y_train)  # fit for the training data
    pred_lin = lin.predict(X_test)  # predict against testing data
    return mean_squared_error(Y_test, pred_lin)  # return the mean squared error of the test data


def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    # Testing Part 2a
    w = fit_LinRegr(X_train, y_train)
    e = mse(X_test, y_test, w)

    # Testing Part 2b
    scikit = test_SciKit(X_train, X_test, y_train, y_test)

    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)


testFn_Part2()

# accuracy of linear regression vs. scikit is identical to 7 decimal places,
# where the float of my version is rounded off.
# Kirill Kresling - 214687735
