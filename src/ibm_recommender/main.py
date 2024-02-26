"""Module main.py"""
import logging
import os
import sys


def main():
    """
    Entry point.
    :return:
    """
    logger.info("IBM Recommender System")
    user = functions.objects.DataReader.from_csv('data/user-item-interactions.csv')
    print(user.analyze())
    print(user.head())


if __name__ == "__main__":
    # Always better to write relative path
    # than absolute path
    # and always best to use the path functions
    # instead of strings
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, "src"))

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s\n%(asctime)s.%(msecs).03d',
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(__name__)

    # Modules
    import functions.objects

    main()
