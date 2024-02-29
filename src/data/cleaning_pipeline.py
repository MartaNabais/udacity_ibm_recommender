import pandas as pd
from sqlalchemy import create_engine


def count_missing_vals(df) -> None:
    """
    This function prints number of missing values for each column
    in pandas dataframe, if missing values are > 0.
    :param df: pandas data frame.
    :return: None.
    """

    nulls_vec = df.count()
    total = df.shape[0]
    for col in df.columns:
        nulls = total - nulls_vec[col]
        if nulls > 0:
            print('There are {} missing values in {}.'.format(nulls, col))


def load_sql_table(df, db_path, table_name) -> None:
    """
    This function takes a pandas data frame, table_name string
    and filepath for the database, and saves the df as a table in the
    SQL database.
    :param df: pandas data frame.
    :param db_path: database filepath.
    :param table_name: table name.
    :return: None.
    """
    conn = create_engine('sqlite:///' + db_path)
    df.to_sql(table_name, con=conn, index=False, if_exists='replace')


def extract_transform_load(user_item_path, articles_path, db_path,
                           table_name_user_item, table_name_content) -> None:
    """
    This function reads and cleans the two initial csv files.
    It also loads the cleaned data frames as tables, in a SQL database.
    :param user_item_path: user-item interactions csv filepath.
    :param articles_path: articles csv filepath
    :param db_path: database filepath.
    :param table_name_user_item: table name for user_item df (str).
    :param table_name_content: table name for articles content df (str).
    :return: None.
    """
    df_user_item = pd.read_csv(user_item_path)
    df_content = pd.read_csv(articles_path)

    del df_user_item['Unnamed: 0']  # Don't need this artefactual columns
    del df_content['Unnamed: 0']

    print('For df_articles: \n')
    count_missing_vals(df_user_item)
    print('For df_content: \n')
    count_missing_vals(df_content)

    # Remove any rows that have the same article_id - only keep the first
    df_content.drop_duplicates(subset=['article_id'], inplace=True)

    # Transform article_id to string
    df_user_item['article_id'] = df_user_item['article_id'].apply(int).apply(str)
    df_content['article_id'] = df_content['article_id'].apply(str)

    # Load dfs to db
    load_sql_table(df_user_item, db_path, table_name_user_item)
    load_sql_table(df_content, db_path, table_name_content)



