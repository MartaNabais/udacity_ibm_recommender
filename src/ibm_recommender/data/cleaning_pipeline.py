import pandas as pd
from sqlalchemy import create_engine


def count_missing_vals(df):
    """
    This function prints number of missing values for each column
    in pandas dataframe, if missing values are > 0.
    :param df:
    :return:
    """

    nulls_vec = df.count()
    total = df.shape[0]
    for col in df.columns:
        nulls = total - nulls_vec[col]
        if nulls > 0:
            print('There are {} missing values in {}.'.format(nulls, col))


def load_sql_table(df, db_path, table_name):
    """
    This function takes a pandas data frame, table_name string
    and filepath for the database, and saves the df as a table in the
    SQL database.
    :param df:
    :param db_path:
    :param table_name:
    :return:
    """
    conn = create_engine('sqlite:///' + db_path)
    df.to_sql(table_name, con=conn, index=False, if_exists='replace')


def extract_transform_load(articles_path, user_item_path, db_path):
    """
    This function reads and cleans the two initial csv files.
    It also loads the cleaned data frames as tables, in a SQL database.
    :param articles_path:
    :param user_item_path:
    :param db_path:
    :return:
    """
    df_articles = pd.read_csv(articles_path)
    df_content = pd.read_csv(user_item_path)

    del df_articles['Unnamed: 0']  # Don't need this artifactual columns
    del df_content['Unnamed: 0']

    print('For df_articles: \n')
    count_missing_vals(df_articles)
    print('For df_content: \n')
    count_missing_vals(df_content)

    # Remove any rows that have the same article_id - only keep the first
    df_content.drop_duplicates(subset=['article_id'], inplace=True)

    # Load dfs to db
    load_sql_table(df_articles, db_path, 'articles_table')
    load_sql_table(df_content, db_path, 'content_table')



