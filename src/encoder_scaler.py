import pandas as pd
from sklearn.preprocessing import StandardScaler



#Frequency encoder and scaling for modelling
def freq_encode_and_scale(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list,
    target_col: str = "DV",
    do_scale: bool = True
):
    """
    Frequency-encodes categorical columns, scales the remaining numeric features
    with StandardScaler, and splits the data back into features & target.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data. Must contain the target column (`target_col`).
    test_df : pd.DataFrame
        Test data. Must contain the target column (`target_col`) - will be
        returned unchanged if present.
    categorical_cols : list
        List of column names that are categorical and should be
        frequency-encoded.
    target_col : str, default "DV"
        Name of the target column.
    do_scale : bool, default True
        If False, the function will only frequency-encode and *not* scale.
        Useful for quick checks or if you want to apply a different scaler.

    Returns
    -------
    X_train : pd.DataFrame
        Scaled feature matrix for training (all columns *except* the target).
    X_test  : pd.DataFrame
        Scaled feature matrix for test.
    y_train : pd.Series
        Target vector for training.
    y_test  : pd.Series
        Target vector for test.

    Notes
    -----
    * The function will create a new `StandardScaler` internally and fit it
      only on the training data. The same scaler is used to transform the
      test set.
    * Frequency maps for each categorical column are built from the training
      set only. Missing categories in the test set are replaced with 0.
    * If you need to reuse the `StandardScaler` or the frequency maps later
      (e.g., for inference on a third dataset), return them from the
      function or store them as attributes of a wrapper class.
    """
    # ------------------------------------------------------------------
    # 1. Pull out the target
    # ------------------------------------------------------------------
    y_train = train_df[target_col].reset_index(drop=True)
    y_test  = test_df[target_col].reset_index(drop=True)

    # Drop target from the feature sets
    X_train_raw = train_df.drop(columns=[target_col]).copy()
    X_test_raw  = test_df.drop(columns=[target_col]).copy()

    # ------------------------------------------------------------------
    # 2. Frequency encode categorical columns
    # ------------------------------------------------------------------
    freq_maps = {}                 # keep for test mapping (and later use)
    X_train_enc = X_train_raw.copy()
    X_test_enc  = X_test_raw.copy()

    for col in categorical_cols:
        # frequency of each category in *train* only
        freq = X_train_raw[col].astype(str).value_counts(normalize=True)
        freq_maps[col] = freq

        X_train_enc[col] = X_train_raw[col].astype(str).map(freq).astype(float)
        # map test, fill unseen categories with 0
        X_test_enc[col]  = X_test_raw[col].astype(str).map(freq).fillna(0).astype(float)

    # ------------------------------------------------------------------
    # 3. Scale numeric features (optional)
    # ------------------------------------------------------------------
    if do_scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_enc)
        X_test_scaled  = scaler.transform(X_test_enc)

        X_train = pd.DataFrame(X_train_scaled, columns=X_train_enc.columns)
        X_test  = pd.DataFrame(X_test_scaled,  columns=X_test_enc.columns)
    else:
        X_train = X_train_enc
        X_test  = X_test_enc

    # ------------------------------------------------------------------
    # 4. Return
    # ------------------------------------------------------------------
    return X_train, X_test, y_train, y_test