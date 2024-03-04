import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame


def compute_similarity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes cosine similarity between each row of a matrix.
    :param df: data frame with rows for which we want to calculate distances.
    :return: a cosine similarity pandas dataframe.
    """
    similarity_df = DataFrame(cosine_similarity(df), index=df.index, columns=df.index)

    return similarity_df


def content_rec(article_id: list, similarity_df: pd.DataFrame,
                df: pd.DataFrame, k=10) -> list:
    """
    This function takes an article id, a similarity matrix, a data frame containing article ids and titles,
    and the number of required recommendations as input.
    It outputs a list of most similar article titles, based on their content.
    :param article_id: article id (str)
    :param similarity_df: similarity matrix (pandas data frame of n x n article ids)
    :param df: pandas data frame of article ids and article titles.
    :param k: number of required recommendations (int)
    :return: a list of article titles with similar content to the input article id.
    """
    similar_articles = list(similarity_df.loc[:, article_id].drop(article_id).sort_values(by=article_id,
                                                                                          ascending=False)[:k].index)
    # To remove potential duplicate article ids, will use set below
    titles = list(set(df[df['article_id'].isin(similar_articles)].doc_full_name))
    titles = list(map(lambda x: x.title(), titles))

    return titles


def email_mapper(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function maps the user email to a user_id column and removes the encoded email column.
    :param df: data frame with user emails.
    :return: updated data frame with mapped user_ids from emails.
    """
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])

    del df['email']
    df.loc[:, 'user_id'] = email_encoded

    return df


def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function returns a pandas data frame with user ids as rows
    and article ids as columns.
    A value of 0 indicates the user has not interacted with the article.
    A value of 1 indicates the user has interacted with the article.
    :param df: pandas data frame with user-article interactions.
    :return: pandas data frame with 0-1 encoded user-article interactions.
    """
    df_copy = df.copy()
    df_copy.loc[:, 'values'] = 1
    user_item = pd.pivot_table(df_copy, values='values', columns='article_id', index='user_id', fill_value=0,
                               dropna=False)
    return user_item
