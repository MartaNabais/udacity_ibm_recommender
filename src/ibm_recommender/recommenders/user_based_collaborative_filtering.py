import pandas as pd


def create_user_item_matrix(df) -> pd.DataFrame:
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
    user_item = pd.pivot_table(df_copy, values='values', columns='article_id',
                               index='user_id', fill_value=0, dropna=False)
    return user_item  # return the user_item matrix


def find_similar_users(user_id, user_item) -> list[str]:
    """
    This function takes a user-article interaction pandas data frame,
    as returned by create_user_item_matrix(), and a user_id, and returns
    a list of the users in order from most to least similar.
    :param user_id: a user id (int)
    :param user_item: a user-article interaction data frame (0-1 values)
    :return: list of the users in order from most to least similar.
    """
    # compute similarity of each user to the provided user
    dot_prod = user_item.loc[user_id, :] @ user_item.transpose()

    # sort by similarity
    dot_prod_sorted = dot_prod.sort_values(ascending=False)

    # create list of just the ids
    most_similar_users = dot_prod_sorted.index.tolist()

    # remove the own user's id
    most_similar_users.remove(user_id)

    return most_similar_users  # return a list of the users in order from most to least similar


def get_article_names(article_ids, df) -> list[str]:
    """
    This function takes a list of article_ids and a pandas dataframe
    with user-article interactions and returns a list of article titles.
    :param article_ids: list of article ids.
    :param df: user-article interaction data frame.
    :return: list of article titles.
    """
    article_names = df[df['article_id'].isin(article_ids)].title.unique().tolist()

    return article_names  # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item, df) -> tuple[list[str], list[str]]:
    """
    This function takes a list of user ids, a user-item interaction pandas data frame
    encoded as 0-1, and a user-item interaction pandas data frame with
    :param user_id:
    :param user_item:
    :param df:
    :return:
    """
    user_articles = user_item.loc[user_id, :]
    article_ids = user_articles[user_articles == 1].index.tolist()
    article_names = get_article_names(article_ids=article_ids, df=df)

    return article_ids, article_names  # return the ids and names


def user_user_recs(user_id, m=10) -> list[str]:
    """
    This function loops through the users based on closeness to the input user id
    For each user it finds the articles the user hasn't seen before and provides
    them as recommendations (recs). Does this until m recommendations are found.
    :param user_id:
    :param m:
    :return:
    """
    similar_users = find_similar_users(user_id=user_id)
    seen_by_user = set(get_user_articles(user_id)[0])
    article_ids = []
    for user_id in similar_users:
        not_seen_article_id = list(seen_by_user.difference(set(get_user_articles(user_id)[0])))
        recs = article_ids + not_seen_article_id
        if len(recs) >= m:
            break

    return recs[:m]  # return your recommendations for this user_id
