from sklearn.linear_model import LinearRegression
import numpy as np
from InvalidDataSetException import InvalidDataSetException
import matplotlib.pyplot as plt


class Criterion(LinearRegression):

    def __init__(self, make_plots=True):
        super().__init__()
        self.make_plots = make_plots
        self.deviations = {}
        self.best_ideal_functions = {}

    def get_best_ideal_functions(self, train_data, ideal_function_data):
        '''
        choosing the ideal functions for the training function is how they minimize the sum of all y-
        deviations squared
        train_data: train data to use
        ideal_function_data: ideal functions to use
        return: a list of indexes of best ideal functions for training function
        '''
        if train_data.shape[1] != 5:
            raise InvalidDataSetException("Train data set should have 5 columns")

        if ideal_function_data.shape[1] != 51:
            raise InvalidDataSetException("Ideal function data set should have 5 columns")

        best_ideal_function_indexes = {}
        for i in range(4):
            train_data_target_index = i + 1
            reg = LinearRegression()
            reg.fit(train_data.x.values.reshape(-1, 1),
                    train_data.iloc[:, train_data_target_index].values.reshape(-1, 1))
            least_square_vals = []

            y_dash = reg.predict(ideal_function_data.x.values.reshape(-1, 1))
            for y in range(50):
                least_square_val = self.calculate_sum_of_least_squares(
                    ideal_function_data.iloc[:, y + 1].values.reshape(-1, 1), y_dash)
                least_square_vals.append(least_square_val)

            best_ideal_function_index = np.argmin(least_square_vals)
            col_name = 'y' + str(best_ideal_function_index + 1)
            self.best_ideal_functions[str(train_data_target_index)] = col_name
            self.deviations[str(train_data_target_index)] = least_square_vals[best_ideal_function_index]

            if self.make_plots:
                self.plot_error_for_best_ideal_function_using_train_data(reg,
                                                                         ideal_function_data,
                                                                         best_ideal_function_index,
                                                                         "Y" + str(i + 1))

        plt.show()
        return self.best_ideal_functions

    def map_test_data(self, ideal_functions_data, test_data_point):
        '''
        maps the test data point to an ideal function so that the calculated regression does not exceed the largest deviation between training dataset and
        the ideal function chosen for it by more than factor sqrt(2)
        ideal_functions_data: ideal function data set
        test_data_point: test data point that is to be mapped
        return: a dictionary object containing the mapped ideal function and calculated deviation for the current test data point
        '''
        test_deviations = {}
        for i in range(4):
            deviation = self.deviations[str(i+1)]
            best_ideal_function = self.best_ideal_functions[str(i+1)]
            ideal_function_model = LinearRegression()
            ideal_function_model.fit(ideal_functions_data.x.values.reshape(-1, 1),ideal_functions_data[best_ideal_function].values.reshape(-1,1))

            y_test_dash = ideal_function_model.predict(test_data_point.x.reshape(-1, 1))
            least_squares_test = self.calculate_sum_of_least_squares(test_data_point.y, y_test_dash)
            if least_squares_test <= (deviation * np.sqrt(2)):
                test_deviations[str(i + 1)] = least_squares_test

        result = {}
        lowest_value = None
        mapped_ideal_function = None
        for key, value in test_deviations.items():
            if lowest_value is None or value < lowest_value:
                lowest_value = value
                mapped_ideal_function = key

        result["deviation"] = lowest_value
        result["ideal_function"] = mapped_ideal_function

        return result

    @staticmethod
    def calculate_sum_of_least_squares(y, y_dash):
        '''
        calculate the sum of least squares between actual and predicted data
        y: actual data
        y_dash: predicted data
        return: a scaler sum of least squares value
        '''
        error = y - y_dash
        least_square_val = np.sum(np.square(error))
        return least_square_val

    @staticmethod
    def plot_error_for_best_ideal_function_using_train_data(model,
                                                            ideal_functions,
                                                            best_ideal_function_index,
                                                            column_name):
        '''
        plots the error for best ideal function found for training dataset
        model: linear regression model trained on training dataset
        ideal_functions: ideal function dataset
        best_ideal_function_index: index of best ideal functions found for training data
        column_name: column name of training dataset(e.g y1,y2) to show in plot title
        return: none
        '''
        y_dash = model.predict(ideal_functions.x.values.reshape(-1, 1))
        y = ideal_functions.iloc[:, best_ideal_function_index + 1].values.reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ideal_functions.x.values, y_dash, "bs", label="predicted values")
        ax.plot(ideal_functions.x.values, y, "k", lw=2, label="actual values")
        ax.legend(loc=3)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f'Error for best ideal function for train data {column_name}')
        plt.savefig(f'plots/{column_name}.png')
