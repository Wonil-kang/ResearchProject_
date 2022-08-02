
def mse(y_test, y_predict):

    length = len(y_test)
    summation = 0
    n = 0

    for i in range(y_test.shape[0]):
        error = y_test[i] - y_predict[i]

        square = error**2
        summation += square
        n += 1

    mse = summation / n

    return mse[0]


def mae(y_test, y_predict):

    length = len(y_test)
    summation = 0
    n = 0

    for i in range(y_test.shape[0]):
        error = y_test[i] - y_predict[i]

        if error < 0:
            error = -error

        summation += error
        n += 1

    mae = summation / n

    return mae[0]