import unittest
from Criterion import Criterion
import pandas as pd
import numpy as np

from InvalidDataSetException import InvalidDataSetException


class MyTestCase(unittest.TestCase):
    def test_get_best_ideal_functions_returns_valid_data(self):
        '''
        Tests if valid data is returned using get_best_ideal_functions function of Criterion class
        '''
        train_data = pd.read_csv("data/train.csv")
        ideal_function_data = pd.read_csv("data/ideal.csv")
        obj_criterion = Criterion(make_plots=False)
        best_ideal_function_indexes = obj_criterion.get_best_ideal_functions(train_data, ideal_function_data)
        self.assertIsNotNone(best_ideal_function_indexes)

    def test_get_best_ideal_functions_returns_correct_data(self):
        '''
        Tests if correct number best ideal functions are returned for train data using get_best_ideal_functions function of Criterion class
        '''
        train_data = pd.read_csv("data/train.csv")
        ideal_function_data = pd.read_csv("data/ideal.csv")
        obj_criterion = Criterion(make_plots=False)
        best_ideal_function_indexes = obj_criterion.get_best_ideal_functions(train_data, ideal_function_data)
        self.assertEqual(len(best_ideal_function_indexes), 4)

    def test_get_best_ideal_functions_invalid_train_data(self):
        '''
        Tests if exception is raised when the train data is invalid
        '''
        train_data = pd.read_csv("data/train.csv")
        ideal_function_data = pd.read_csv("data/ideal.csv")
        train_data['y5'] = train_data.y2
        obj_criterion = Criterion(make_plots=False)
        with self.assertRaises(InvalidDataSetException):
            best_ideal_function_indexes = obj_criterion.get_best_ideal_functions(train_data, ideal_function_data)

    def test_get_best_ideal_functions_invalid_ideal_function_data(self):
        '''
        Tests if exception is raised when the ideal function data is invalid
        '''
        train_data = pd.read_csv("data/train.csv")
        ideal_function_data = pd.read_csv("data/ideal.csv")
        ideal_function_data['y51'] = ideal_function_data.y50
        obj_criterion = Criterion(make_plots=False)
        with self.assertRaises(InvalidDataSetException):
            best_ideal_function_indexes = obj_criterion.get_best_ideal_functions(train_data, ideal_function_data)

    def test_calculate_sum_of_least_squares(self):
        '''
        Tests if the calculate_sum_of_least_squares calculates the right value
        '''
        y = np.array([1, 2, 5, 8])
        y_dash = np.array([1, 3, 6, 7])
        least_square = Criterion(make_plots=False).calculate_sum_of_least_squares(y, y_dash)
        self.assertEqual(least_square, 3)


if __name__ == '__main__':
    unittest.main()
