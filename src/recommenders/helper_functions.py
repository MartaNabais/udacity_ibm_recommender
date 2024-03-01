from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame


def compute_similarity_df(df):
    """
    This function computes cosine similarity between each row of a matrix.
    :param df: data frame with rows for which we want to calculate distances.
    :return: a cosine similarity pandas dataframe.
    """
    similarity_df = DataFrame(cosine_similarity(df), index=df.index, columns=df.index)

    return similarity_df


def content_rec(article_id, similarity_df, df, title_column, k=10) -> list:
    """
    This function takes an article id, a similarity matrix, a data frame containing article ids and titles,
    and the number of required recommendations as input.
    It outputs a list of most similar article titles, based on their content.
    :param article_id: article id (str)
    :param similarity_df: similarity matrix (pandas data frame of n x n article ids)
    :param df: pandas data frame of article ids and article titles.
    :param title_column: column containing article titles (str)
    :param k: number of required recommendations (int)
    :return: a list of article titles with similar content to the input article id.
    """

    similar_articles = list(similarity_df.loc[:, article_id].drop(article_id).sort_values(ascending=False)[:k].index)
    # To remove potential duplicate article ids, will use set below
    titles = list(set(df[df['article_id'].isin(similar_articles)][title_column]))
    titles = list(map(lambda x: x.title(), titles))

    return titles
