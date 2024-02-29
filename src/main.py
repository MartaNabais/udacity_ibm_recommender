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
    data.cleaning_pipeline.extract_transform_load(args.user_item_csv,
                                                  args.article_content_csv,
                                                  args.db_filepath,
                                                  args.user_item_table,
                                                  args.article_content_table)

    df = data.reader.read_db(args.db_filepath, args.user_item_table)
    df_content = data.reader.read_db(args.db_filepath, args.article_content_table)
    logger.info('')
    print(df.head())
    print(df_content.head())


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

    parser.add_argument('user_item_csv')
    parser.add_argument('article_content_csv')
    parser.add_argument('db_filepath')
    parser.add_argument('user_item_table')
    parser.add_argument('article_content_table')

    args = parser.parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s\n%(asctime)s.%(msecs).03d',
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(__name__)

    # Modules
    import data.reader
    import data.cleaning_pipeline

    main()
