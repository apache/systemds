import pandas as pd
from sklearn import linear_model
from sklearn.datasets import make_regression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def generate_dataset(beta, n):
    # Generate x as an array of `n` samples which can take a value between 0 and 5
    x = np.random.random_integers(1, 3, (n, 3))
    # Calculate `y` according to the equation discussed
    coeff = [2, 3, 1]
    coeff_mat = np.array(coeff).reshape(len(coeff), 1)
    y = np.matmul(x, coeff_mat)
    dataset = np.append(x, y, axis=1)
    np.savetxt("toy_train.csv", dataset, delimiter=",", fmt='%s')
    return x, y


if __name__ == '__main__':
    def main():
        x, y = generate_dataset(10, 100)
        x_train = x
        y_train = y
        dataset = np.append(x, y, axis=1)
        np.savetxt("toy_train.csv", dataset, delimiter=",", fmt='%s')
        model = linear_model.LinearRegression()

        # Train the model using the training data that we created
        model.fit(x_train, y_train)
        # Now that we have trained the model, we can print the coefficient of x that it has predicted
        print('Coefficients: \n', model.coef_)

        # We then use the model to make predictions based on the test values of x
        test_dataset = pd.read_csv("../datasets/toy.csv")
        attributes_amount = len(test_dataset.values[0])
        y_test = test_dataset.iloc[:, attributes_amount - 1:attributes_amount].values
        x_test = test_dataset.iloc[:, 0:attributes_amount - 1].values
        y_pred = model.predict(x_test)

        # Now, we can calculate the models accuracy metrics based on what the actual value of y was
        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, y_pred))
        print('r_2 statistic: %.2f' % r2_score(y_test, y_pred))
        # Coefficients: [[2. 3. 1.]]
        # Mean squared error: 3.00
        # r_2 statistic: 0.72

        # dataset = np.append(x, y, axis=1)
        # np.savetxt("toy.csv", dataset, delimiter=",", fmt='%s')

        return model, x_test, y_test, y_pred




