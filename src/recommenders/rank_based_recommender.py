

def get_top_article_ids(df, n=10) -> list:
    """
    This function returns the top n articles indices the users have interacted with.
    :param n: number of articles to return (int)
    :param df: pandas dataframe with article-users interactions
    :return: list of top articles indices
    """
    article_counts = df.article_id.value_counts()
    top_articles_idx = list(article_counts[:n,].index)

    return top_articles_idx


def get_top_articles(df, n=10) -> list:
    """
    This function returns the top n articles titles the users have interacted with.
    :param n: number of articles to return (int)
    :param df: pandas dataframe with article-users interactions
    :return: list of top articles titles
    """
    top_idx = get_top_article_ids(n=n, df=df)
    top_articles = list(df[df['article_id'].isin(top_idx)].title.unique())

    return top_articles
