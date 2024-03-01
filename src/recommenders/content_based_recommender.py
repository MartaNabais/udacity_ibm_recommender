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
    vectorizer = TfidfVectorizer(stop_words='english',
                                 token_pattern=r'(?u)\b[A-Za-z]+\b',
                                 ngram_range=(1, 1))
    # x is a TF-IDF vectorized array
    x = vectorizer.fit_transform(articles_df['doc_full_name'].values.astype('U'))
    articles_idx = articles_df['article_id']  # get article ids, to use as column names
    count_vec_df = pd.DataFrame(x.toarray(),
                                columns=vectorizer.get_feature_names_out(),
                                index=articles_idx)

    return count_vec_df
