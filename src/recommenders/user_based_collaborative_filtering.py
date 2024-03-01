import pandas as pd


def get_article_names(article_ids, df) -> list[str]:
    """
    This function takes a list of article_ids and a pandas dataframe
    with user-article interactions and returns a list of article titles.
    :param article_ids: article ids (list).
    :param df: user-item interaction (data frame).
    :return: list of article titles.
    """
    article_names = df[df['article_id'].isin(article_ids)].title.unique().tolist()

    return article_names  # Return the article names associated with list of article ids


def get_user_articles(user_id, df, user_item) -> tuple[list[str], list[str]]:
    """
    This function takes user id, a user-item interaction data frame,
     and a user-item interaction pandas data frame with values encoded as 0-1,
     and returns a tuple of a list with article ids and article names for the user.
    :param user_id: user id (int).
    :param df: data frame of user-item interactions (long-format)
    :param user_item: data frame of user-item interactions (wide-format,
    items in columns) encoded as 0 (if user has not interacted with article)-1 (if user interacted with article)
    :return: tuple with a list of article ids and a list of article names.
    """
    user_articles = user_item.loc[user_id, :]
    article_ids = user_articles[user_articles == 1].index.tolist()
    article_names = get_article_names(article_ids=article_ids, df=df)

    return article_ids, article_names


def get_top_sorted_users(user_id, user_item) -> pd.DataFrame:
    """
    This function takes a user id and a data frame of
    user-item interactions (long-format, items in columns), with
    0 meaning a user has not interacted with article and 1 meaning
    the opposite and outputs the most similar users, sorted by level
    of similarity and number of interactions.
    :param user_id: user id (int)
    :param user_item: user-item interactions (data frame)
    :return: data frame with top sorted users.
    """
    dot_prod = user_item.loc[user_id, :] @ user_item.transpose()

    # sort by similarity
    dot_prod_sorted = dot_prod.sort_values(ascending=False)

    # create list of just the ids
    most_similar_users = dot_prod_sorted.index.tolist()

    # remove the own user's id
    most_similar_users.remove(user_id)  # this is a list so need to use 'remove'
    similarity = dot_prod_sorted.drop(user_id).values  # this is a data frame so need to use 'drop'
    num_interactions = user_item.drop(user_id).sum(axis=1).values

    neighbors_df = pd.DataFrame({'neighbor_id': most_similar_users,
                                 'similarity': similarity,
                                 'num_interactions': num_interactions})

    neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False)

    return neighbors_df


def user_user_recs(user_id, df, user_item, k=10) -> tuple[list[str], list[str]]:
    """
    This function takes a user id and outputs the number of recommendations
    required by m, taking into account the similarity to other users and
    the number of interactions the users have had (more similar and more interactive
    are at the top).
    :param user_id: user id (int)
    :param df: user-item interaction data frame (wide-format)
    :param user_item: user-item interaction data frame (long-format; items as columns)
    :param m: number of recommendations required (int)
    :return: tuple with list of recommended article ids, and list of recommended article titles
    """
    similar_users = get_top_sorted_users(user_id, user_item=user_item)
    seen_by_user = set(get_user_articles(user_id, df=df, user_item=user_item)[0])
    recs = []
    for other_user_id in similar_users['neighbor_id']:
        seen_by_other_user = set(get_user_articles(other_user_id, df=df, user_item=user_item)[0])
        not_seen_article_id = list(seen_by_other_user.difference(seen_by_user))
        recs = list(set(recs + not_seen_article_id))  # to make sure duplicates are removed
        if len(recs) >= k:
            break

    recs = recs[:k]
    rec_names = get_article_names(recs, df=df)

    return recs, rec_names
