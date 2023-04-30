import pandas as pd
from Criterion import Criterion
from Database import Database
import sys
import traceback
import time


def main():
    '''
    main function to run the full program
    '''
    try:
        train_data = pd.read_csv("data/train.csv")
        ideal_function_data = pd.read_csv("data/ideal.csv")
        test_data = pd.read_csv("data/test.csv")

        obj_criterion = Criterion()
        obj_database = Database()

        obj_database.create_tables()
        obj_database.insert_data_in_tables(train_data, test_data, ideal_function_data)

        best_ideal_functions = obj_criterion.get_best_ideal_functions(train_data, ideal_function_data)

        for index, row in test_data.iterrows():
            result = obj_criterion.map_test_data(ideal_function_data, row)

            if result["deviation"] is None or result["ideal_function"] is None:
                continue

            obj_database.update_test_data_point(row, result["deviation"], result["ideal_function"])


    except:
        exception_info = get_exception_info()
        print(exception_info)
    finally:
        print("Program execution completed!")


def get_exception_info():
    try:
        exception_type, exception_value, exception_traceback = sys.exc_info()
        file_name, line_number, procedure_name, line_code = traceback.extract_tb(exception_traceback)[-1]
        exception_info = ''.join('[Time Stamp]: '
                                 + str(time.strftime('%d-%m-%Y %I:%M:%S %p'))
                                 + '' + '[File Name]: ' + str(file_name) + ' '
                                 + '[Procedure Name]: ' + str(procedure_name) + ' '
                                 + '[Error Message]: ' + str(exception_value) + ' '
                                 + '[Error Type]: ' + str(exception_type) + ' '
                                 + '[Line Number]: ' + str(line_number) + ' '
                                 + '[Line Code]: ' + str(line_code))

        return exception_info
    except:
        pass

if __name__ == '__main__':
    main()
