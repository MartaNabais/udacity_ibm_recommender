import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def intersect_articles(df, df_content) -> pd.DataFrame():
    """
    This function takes the user-item interaction and article-content data frames
    and checks which article ids in the former are not present in the latter.
    Then it concatenates the two data frames together (binding rows), to add missing
    article ids.
    :param df: user-item original data frame.
    :param df_content: article content original data frame.
    :return: a merged pandas data frame, with article ids in user-item data frame that are not in
    article-content data frame..
    """

    df_new = df[~df['article_id'].isin(df_content['article_id'])]
    df_new.drop_duplicates(subset='article_id', inplace=True)
    df_new.rename(columns={"title": "doc_full_name"}, inplace=True)

    df_merged = pd.concat([df_new[['article_id', 'doc_full_name']],
                           df_content[['article_id', 'doc_full_name']]], sort=False)
    return df_merged


def tfidf_vectorizer(articles_df) -> pd.DataFrame():
    """
    This function takes a data frame with article titles and article ids as input.
    It performs TF-IDF vectorization of the article title contents.
    :param articles_df:
    :return: a pandas data frame with a TF-IDF array (with words as columns and article ids as rows).
    """
    # Initiated vectorizer object
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[A-Za-z]+\b', ngram_range=(1, 1))
    # x is a TF-IDF vectorized array
    x = vectorizer.fit_transform(articles_df['doc_full_name'].values.astype('U'))
    articles_idx = articles_df['article_id']  # get article ids, to use as column names
    count_vec_df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out(), index=articles_idx)

    return count_vec_df


def content_rec(article_id, similarity_df, df, k=10) -> list:
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

    similar_articles = list(similarity_df.loc[:, article_id].drop(article_id).sort_values(ascending = False)[:k].index)
    titles = list(set(df[df['article_id'].isin(similar_articles)].doc_full_name))
    titles = list(map(lambda x: x.title(), titles))

    return titles
