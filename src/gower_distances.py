import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
import gower #For distances when variables are categorical and continuos
import math


#Gower distalnces
def gower_closest_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_col: Union[str, list[str], None] = None,
    cluster_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return a subset of the training set that is *closest* to the test set
    according to the Gower distance.

    Parameters
    ----------
    train : pd.DataFrame
        The full training set.  Must have a row-index that can be used to
        identify the selected rows.
    test : pd.DataFrame
        The test set to which we compute distances.  Only the *feature*
        columns are used; the target columns are dropped automatically.
    target_col : str or list[str], optional
        Name(s) of the column(s) that hold the target / label.
        They are excluded from the distance calculation.
        If ``None`` the last column of each dataframe is assumed to be the
        target (matching the original code that used ``iloc[:, :-1]``).
    cluster_size : int, optional
        Number of nearest training rows to keep.
        If ``None`` it defaults to ``ceil(sqrt(len(train)))``.

    Returns
    -------
    pd.DataFrame
        A *slice* of ``train`` that contains the ``cluster_size`` rows
        with the smallest Gower distance to the test set.
    """
    # ------------------------------------------------------------------
    # 1. Prepare the feature matrices
    # ------------------------------------------------------------------
    if target_col is None:
        # Assume the last column is the target
        train_features = train.iloc[:, :-1]
        test_features = test.iloc[:, :-1]
    else:
        # Drop the target column(s) if supplied
        if isinstance(target_col, str):
            target_col = [target_col]
        train_features = train.drop(columns=target_col)
        test_features = test.drop(columns=target_col)

    # ------------------------------------------------------------------
    # 2. Compute the Gower distance matrix
    # ------------------------------------------------------------------
    # gower.gower_matrix accepts pandas DataFrames directly
    gower_dist = gower.gower_matrix(train_features, test_features)
    dist_df = pd.DataFrame(gower_dist)           # rows → train, columns → test
    train_idx = pd.Series(train.index, name="train_index")

    # ------------------------------------------------------------------
    # 3. Find the *closest* training rows to the first test instance
    # ------------------------------------------------------------------
    # Only the distance to the *first* test instance matters for the cluster
    # (you can modify this if you need a different strategy).
    distances = pd.concat([train_idx, dist_df.iloc[:, 0:1]], axis=1)
    distances.columns = ["train_index", "dist"]

    # Sort by distance
    closest = distances.nsmallest(
        cluster_size or math.ceil(math.sqrt(len(train))), "dist"
    )

    # Grab the indices
    selected_indices = closest["train_index"]

    # Return the subset of training data
    return train.loc[selected_indices].copy()
