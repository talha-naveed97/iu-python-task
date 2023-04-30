import pymysql
import sqlalchemy as db
from DatabaseOperationException import DatabaseOperationException
from InvalidDataSetException import InvalidDataSetException


class Database():

    def __init__(self):
        self.tbl_train = "train_data"
        self.tbl_test = "test_data"
        self.tbl_ideal = "ideal_function"
        self.engine = db.create_engine("mysql+pymysql://root:test123@localhost/pythonDB")
        self.meta_data = db.MetaData()
        self.connection = None

    def connect(self):
        '''
        connect to database
        '''
        self.connection = self.engine.connect()

    def dispose(self):
        '''
        dispose connection to database
        '''
        self.engine.dispose()

    def create_tables(self):
        '''
        create all the required tables i.e. train data, test data, ideal functions
        '''
        try:
            self.connect()
            try:
                train_data_tbl = db.Table(self.tbl_train, self.meta_data, autoload=True, autoload_with=self.engine)
                test_data_tbl = db.Table(self.tbl_test, self.meta_data, autoload=True, autoload_with=self.engine)
                ideal_function_tbl = db.Table(self.tbl_ideal, self.meta_data, autoload=True, autoload_with=self.engine)
                self.meta_data.drop_all(self.engine)
            except:
                pass

            self.meta_data = db.MetaData()
            train_data_tbl = db.Table(
                self.tbl_train, self.meta_data,
                db.Column("X", db.DECIMAL(10, 2), nullable=False),
                db.Column("Y1", db.DECIMAL(10, 2), nullable=False),
                db.Column("Y2", db.DECIMAL(10, 2), nullable=False),
                db.Column("Y3", db.DECIMAL(10, 2), nullable=False),
                db.Column("Y4", db.DECIMAL(10, 2), nullable=False))

            test_data_tbl = db.Table(
                self.tbl_test, self.meta_data,
                db.Column("X", db.DECIMAL(10, 2), nullable=False),
                db.Column("Y", db.DECIMAL(10, 2), nullable=False),
                db.Column("DeltaY", db.DECIMAL(10, 2), nullable=True),
                db.Column("NoOfIdealFunc", db.Integer, nullable=True))

            columns = list()
            columns.append(db.Column("X", db.DECIMAL(10, 2), primary_key=False))

            for i in range(50):
                col_name = "Y" + str(i + 1)
                columns.append(db.Column(col_name, db.DECIMAL(10, 2), primary_key=False))

            ideal_function_tbl = db.Table(self.tbl_ideal, self.meta_data, *columns)

            self.meta_data.create_all(self.engine)

        except:
            raise DatabaseOperationException("Error while creating datatables in database")
        finally:
            self.dispose()

    def insert_data_in_tables(self, train_data, test_data, ideal_function_data):
        '''
        inserts all data rows that were in dataset to database tables for train data, test data and ideal functions
        '''
        try:
            if train_data.shape[1] != 5:
                raise InvalidDataSetException("Train data set should have 5 columns")

            if ideal_function_data.shape[1] != 51:
                raise InvalidDataSetException("Ideal function data set should have 5 columns")

            if test_data.shape[1] != 2:
                raise InvalidDataSetException("Ideal function data set should have 5 columns")

            self.connect()
            self.meta_data = db.MetaData()
            train_data_tbl = db.Table(self.tbl_train, self.meta_data, autoload=True, autoload_with=self.engine)
            sql_query = db.insert(train_data_tbl)

            data_list = []
            for index, row in train_data.iterrows():
                data_obj = {"X": row['x'], "Y1": row['y1'], "Y2": row['y2'],
                            "Y3": row['y3'], "Y4": row['y4']}
                data_list.append(data_obj)

            self.connection.execute(sql_query, data_list)

            data_list = []
            test_data_tbl = db.Table(self.tbl_test, self.meta_data, autoload=True, autoload_with=self.engine)
            sql_query = db.insert(test_data_tbl)

            for index, row in test_data.iterrows():
                data_obj = {"X": row['x'], "Y": row['y'], "DeltaY": None,
                            "NoOfIdealFunc": None}
                data_list.append(data_obj)

            self.connection.execute(sql_query, data_list)

            ideal_data_tbl = db.Table(self.tbl_ideal, self.meta_data, autoload=True, autoload_with=self.engine)
            sql_query = db.insert(ideal_data_tbl)

            data_list = []
            for index, row in ideal_function_data.iterrows():
                data_obj = {"X": row['x']}
                for i in range(50):
                    col_name = "Y" + str(i + 1)
                    row_col = "y" + str(i + 1)
                    data_obj[col_name] = row[row_col]

                data_list.append(data_obj)

            self.connection.execute(sql_query, data_list)

        except:
            raise DatabaseOperationException("Error while inserting data in datatables")
        finally:
            self.dispose()

    def update_test_data_point(self, test_data_point, deviation, ideal_function):
        '''
        update deltaY and NoOfIdealFunc columns for test data table in database
        '''
        try:
            self.connect()
            self.meta_data = db.MetaData()
            test_data_tbl = db.Table(self.tbl_test, self.meta_data, autoload=True, autoload_with=self.engine)
            sql_query = db.update(test_data_tbl).values(DeltaY=deviation,NoOfIdealFunc=ideal_function).where(test_data_tbl.columns.X == test_data_point.x and test_data_tbl.columns.Y == test_data_point.y)
            self.connection.execute(sql_query)
        except:
            raise DatabaseOperationException("Error while inserting data in datatables")
        finally:
            self.dispose()