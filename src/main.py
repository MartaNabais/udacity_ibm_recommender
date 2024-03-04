"""Module main.py"""
import logging
import os
import sys
import argparse
from sklearn.metrics.pairwise import cosine_similarity


def main():
    """
    Entry point.
    :return:
    """
    logger.info('IBM Recommender System')

    if args.etl_pipeline == 'Y':
        logger.info('Reading and cleaning data.')
        data.cleaning_pipeline.extract_transform_load('src/data/user-item-interactions.csv',
                                                      'src/data/articles_community.csv',
                                                      'src/data/ibm_recommender.db',
                                                      'user_item_table',
                                                      'article_content_table')

    df_articles = data.reader.read_db('src/data/ibm_recommender.db', 'user_item_table')
    df_articles = recommenders.helper_functions.email_mapper(df=df_articles)
    df_content = data.reader.read_db('src/data/ibm_recommender.db', 'article_content_table')

    # Will use for content-based recommendations
    articles_df = recommenders.content_based_recommender.intersect_articles(df=df_articles, df_content=df_content)
    count_vect_df = recommenders.content_based_recommender.tfidf_vectorizer(articles_df=articles_df)
    articles_similarity_df = recommenders.helper_functions.compute_similarity_df(df=count_vect_df)

    if args.user_id not in df_articles['user_id']:
        top_articles = recommenders.rank_based_recommender.get_top_articles(df=df_articles)
        logger.info('Hello {}!'.format(args.user_id))
        print('Here are the top 10 articles we recommend for you: \n')
        [print(article.title()) for article in top_articles]

        # Content-based recommendation for cold-start
        top_article_id = recommenders.rank_based_recommender.get_top_article_ids(df=df_articles, n=1)
        recs = recommenders.helper_functions.content_rec(article_id=top_article_id,
                                                         similarity_df=articles_similarity_df,
                                                         df=articles_df)
        print('\n You may also like: \n')
        [print(article.title()) for article in recs]

    else:
        logger.info('Hello {}!'.format(args.user_id))
        # User-based collaborative filtering
        logger.info('Performing user-based recommendations.')
        user_item = recommenders.helper_functions.create_user_item_matrix(df=df_articles)
        recs = recommenders.user_based_collaborative_filtering.user_user_recs(user_id=args.user_id,
                                                                              df=df_articles,
                                                                              user_item=user_item)[1]
        print('Based on your viewing preferences we recommend: \n')
        [print(article.title()) for article in recs]

        # SVD-based recommendations
        # logger.info('Performing SVD-based recommendations.')
        # svd_df = recommenders.svd_recommender.get_vt(user_item=user_item)
        # svd_similarity_df = recommenders.helper_functions.compute_similarity_df(df=svd_df.T)
        # recs = recommenders.helper_functions.content_rec(article_id=args.user_id,
        #                                                  similarity_df=svd_similarity_df,
        #                                                  df=df_articles)[1]
        # logger.info('SVD-based recommendation for user {}: \n {}'.format(args.user_id, recs))

        #
        # Content based recommendation
        logger.info('Performing content-based recommendations.')
        article_id = list(df_articles[df_articles['user_id'] == args.user_id].article_id.unique())
        recs = recommenders.helper_functions.content_rec(article_id=article_id,
                                                         similarity_df=articles_similarity_df,
                                                         df=articles_df)
        watched = ''.join(recommenders.user_based_collaborative_filtering.get_article_names(article_id, df=df_articles))
        watched = watched.title()
        print('Because you watched: {} \n'.format(watched))
        print('You may also like:')
        [print(article.title()) for article in recs]


if __name__ == "__main__":
    # Always better to write relative path
    # than absolute path
    # and always best to use the path functions
    # instead of strings
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, "src"))

    parser = argparse.ArgumentParser(
        prog='IBM Recommender System',
        description='This program creates recommender system for IBM files.',
        epilog='Text at the bottom of help')

    parser.add_argument('user_id', type=int, help='User ID.')
    parser.add_argument('etl_pipeline', default='Y', type=str,
                        help='Perform ETL pipeline? Y or N.')
    parser.add_argument('--article_id', type=str,
                        help='If content-based recommendation required, please provide article ID.')

    args = parser.parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s\n%(asctime)s.%(msecs).03d',
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(__name__)

    # Modules
    import data.reader
    import data.cleaning_pipeline
    import recommenders.content_based_recommender
    import recommenders.helper_functions
    import recommenders.svd_recommender
    import recommenders.rank_based_recommender
    import recommenders.user_based_collaborative_filtering

    main()
