import numpy as np
import pandas as pd


def get_vt(user_item: pd.DataFrame, k=100) -> pd.DataFrame:
    """
    This function performs Singular Value Decomposition on a user-item interaction
    matrix of 0-1 values and returns the diagonal right singular matrix, indicating
    similarity between items and latent features.
    :param user_item: user-item interaction data frame.
    :param k: number of latent features.
    :return: the diagonal right singular matrix, indicating similarity between
     items and latent features.
    """
    # Get u, s and v transpose matrices
    u, s, vt = np.linalg.svd(user_item)

    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]

    vt_new = pd.DataFrame(vt_new, columns=user_item.columns)

    return vt_new

