from sqlalchemy import create_engine
import pandas as pd


def read_db(db_filepath, table_name):
    """
    This function reads all columns in the table
    given by the table name and database given by the db_filepath.

    :param db_filepath: relative filepath to database.
    :param table_name: table name.
    :return:
    """
    conn = create_engine('sqlite:///' + db_filepath)
    df = pd.read_sql_table(table_name, conn)

    return df
