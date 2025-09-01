import pandas as pd



def missing(
        dataset: pd.DataFrame,
):
    # Count total missing values
    missing_count = dataset.isna().sum().sum()
    print("Total missing datapoints:", missing_count)
    # Get dataset dimensions: number of rows and columns
    n_rows, n_cols = dataset.shape
    print("Dataset dimensions: {} rows x {} columns".format(n_rows, n_cols))
    # Compute total number of cells in the dataset
    total_cells = n_rows * n_cols
    # Calculate percentage of missing data
    missing_percentage = (missing_count / total_cells) * 100
    print("Percentage of missing data: {:.2f}%".format(missing_percentage))

    """
    Returns the number of missing data in the dataframe
    ----------
    """
