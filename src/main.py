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
    logger.info('Reading and cleaning data.')
    data.cleaning_pipeline.extract_transform_load('src/data/user-item-interactions.csv',
                                                  'src/data/articles_community.csv',
                                                  'src/data/ibm_recommender.db',
                                                  'user_item_table',
                                                  'article_content_table')

    df_articles = data.reader.read_db('src/data/ibm_recommender.db', 'user_item_table')
    print(df_articles.head())
    df_content = data.reader.read_db('src/data/ibm_recommender.db', 'article_content_table')

    # Content based recommendation
    logger.info('Performing content-based recommendations.')
    articles_df = recommenders.content_based_recommender.intersect_articles(df=df_articles, df_content=df_content)
    count_vect_df = recommenders.content_based_recommender.tfidf_vectorizer(articles_df=articles_df)
    articles_similarity_df = recommenders.helper_functions.compute_similarity_df(df=count_vect_df)
    recs = recommenders.helper_functions.content_rec(article_id=args.user_id,
                                                     similarity_df=articles_similarity_df,
                                                     df=articles_df)
    logger.info('Content-based recommendation for user {}: \n {}'.format(args.user_id, recs))

    logger.info('Performing SVD-based recommendations.')
    # SVD-based recommendations
    user_item = recommenders.helper_functions.create_user_item_matrix(df=df_articles)
    svd_df = recommenders.svd_recommender.get_vt(user_item=user_item)
    svd_similarity_df = recommenders.helper_functions.compute_similarity_df(df=svd_df.T)
    print(svd_similarity_df)
    recs = recommenders.helper_functions.content_rec(article_id=args.user_id,
                                                     similarity_df=svd_similarity_df,
                                                     df=articles_df)
    logger.info('SVD-based recommendation for user {}: \n {}'.format(args.user_id, recs))


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

    parser.add_argument('user_id')

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

    main()
