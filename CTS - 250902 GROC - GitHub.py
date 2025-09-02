from tkinter.filedialog import askopenfilename
import pandas as pd
import time
import sklearn.ensemble
import numpy as np
import os
from datetime import datetime

#For plotting and missing data
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl #To reset plotting parameters == mpl.rcdefaults()

#To tabulate data
from tabulate import tabulate

#To impute data
import miceforest as mf #To impute categorical and continuous variables

#For modelling
from sklearn.metrics import mean_absolute_error  # Change the import
from sklearn.tree import plot_tree
from sklearn.metrics import mean_absolute_error
from scipy import stats # For t-test

#Custom modules
from src import missing_data, gower_distances, encoder_scaler, models, plotting
from src.data_config import columns_to_retain, categorical_cols, columns_to_check, continuous_cols, final_predictors, CONFIG


# Set the seed for reproducibility
np.random.seed(123)
sklearn.utils.check_random_state(123)


#------------- LOADING DATA ------------------#

filename = askopenfilename()
data = pd.read_excel(io=filename) #For the oversamples
print(data.columns)
print(data.info())
print(data.describe())

# Variables
col_names = data.columns
print(col_names)



#------------- MISSING DATA ANALYSIS & DATA IMPUTATION ------------------#

# Missing datapoints
#Number of missing data, missing data plot, number of participant, and dataset dimensions (# rows and columns).
    #Missing data oversampled dataset
data.isna().sum().sum()
sns.heatmap(data.isna(), cmap='Blues', cbar=True)
plt.show()
        #number of rows
data.shape[0]
data.dropna(subset=['GROC24Adjusted'],inplace=True) #Dropping rows that don't have GROC

# Missing datapoints
#Number of missing data, missing data plot, number of participant, and dataset dimensions (# rows and columns).
    #Missing data oversampled dataset
data.isna().sum().sum()
sns.heatmap(data.isna(), cmap='Blues', cbar=True)
plt.show()
        #number of rows
data.shape[0]

#Prints the number of missing data
missing_data.missing(dataset=data)

# Make categorical variable categorical and continuous variable float
dtype_map = {}
dtype_map.update({col: 'category' for col in categorical_cols})
dtype_map.update({col: 'float64' for col in continuous_cols})
# Assign dtypes in one go
data = data.astype(dtype_map)

#Descriptive statistics
describe_df = data.describe()
# Display as Markdown table (Jupyter or export)
print(tabulate(describe_df, headers='keys', tablefmt='github'))

#Frequency table
# Loop through each column and compute value counts
for col in columns_to_check:
    print(f"\n### Frequency Table for '{col}':")
    freq_table = data[col].value_counts().reset_index()
    freq_table.columns = ['Value', 'Frequency']
    print(tabulate(freq_table, headers='keys', tablefmt='github'))

#Imputing missing data using the MICE for categorical and continuous variables.
# Reset index
data = data.reset_index(drop=True)
# Create a MICE kernel
kernel = mf.ImputationKernel(data,random_state=CONFIG["random_state"])
# Run the MICE imputation process with # of iterations and dataset
kernel.mice(iterations=5)
# Extract imputed dataset
data_imputed = kernel.complete_data(0)

# Missing datapoints
#Number of missing data, missing data plot, number of participant, and dataset dimensions (# rows and columns).
    #Missing data oversampled dataset
data_imputed.isna().sum().sum()
sns.heatmap(data_imputed.isna(), cmap='Blues', cbar=True)
plt.show()
        #number of rows
data_imputed.shape[0]



#------------- FEATURE SELECTION & CORRELATION ANALYSIS ------------------#
#ENCODE CATEGORICAL FOR SNR (Frequency encoding)
# Store the responder column
responder = data_imputed['GROC24Adjusted']
# Store frequency mappings per column
freq_maps = {}
# Frequency encode categorical variables in training data
data_imputed_encoded = data_imputed.copy()
for col in categorical_cols:
    # Calculate frequency of each category
    freq = data_imputed[col].value_counts(normalize=True)
    # Store it for test mapping
    freq_maps[col] = freq 
    print(freq)
    # Replace categories with their frequencies
    data_imputed_encoded[col] = data_imputed[col].map(freq)
# Convert each column to continuos after encoding
data_imputed_encoded[categorical_cols] = data_imputed_encoded[categorical_cols].astype('float64')


# Create a new DataFrame with only included columns
df_included = data_imputed_encoded[columns_to_retain]
# Calculate the correlation matrix
correlation_matrix = df_included.corr(method='spearman')
# Mask upper triangle and diagonal correlations
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
# Improve the plot by adjusting the size and rotating labels
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", annot_kws={"size": 8}, mask=mask)
# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
# Show the plot
plt.show()


# CORRELATION Signal to noise ration (SNR)
    #Creates matrix and then dataframe
SigNoise = np.zeros((1, df_included.shape[1]-1))
SigNoise = pd.DataFrame(SigNoise)
SigNoise.columns = df_included.columns[0:CONFIG["numb_var_assess"]]

for i in range(SigNoise.shape[1]):
    x = correlation_matrix.iloc[len(correlation_matrix)-1,i]
    SigNoise.iloc[0, i] = x
del i


# Ordering data_imputed_over columns based on SNR values
SigNoise = SigNoise.sort_values(by=0, axis=1, ascending=False, key=lambda x: abs(x))
    #Substetting SigNoise to number of variables
trial = SigNoise.iloc[:, :CONFIG["numb_var_assess"]]
    #Saving variables names
PlotXnames = trial.columns
y = trial.iloc[0, :].values #Absolute values


#Line plot
ax = plotting.line_plot(
        y,
        PlotXnames,
        x_label="",
        y_label="",
        dpi=100,           
        figsize=(4, 2)     
    )
plt.ylabel('Correlation value', fontsize=10)
ax.figure.tight_layout()
plt.show()


#Bar plot
ax = plotting.bar_plot(
        y,
        PlotXnames,
        x_label="",
        y_label="Correlation value",
        dpi = 300
    )
plt.ylabel('Correlation value', fontsize=4)
ax.figure.tight_layout()
plt.show()



#Print the top 10 variables based on SNR
print(trial.columns[0:len(PlotXnames)])


#Create new dataset which retains only most important predictors and dependent variable
nomi = PlotXnames.insert(len(PlotXnames), 'GROC24Adjusted') 
data = data_imputed[nomi]
#Data set up for modelling
data = data.rename(columns={'GROC24Adjusted': 'DV'})
accuracy_columns = ["Sample Size", "Actual", "MAE", "Predicted"]
accuracy = pd.DataFrame(np.zeros((data.shape[0], len(accuracy_columns))), columns=accuracy_columns)



#------------- MODELLING PIPELINE ------------------#
beginning = time.time()

for i in range(0,len(data)):
    print(i)

    test = data.iloc[i:i+1,:]
    train = data.drop(i, axis=0)
    
    train_fixed = train.copy()
    test_fixed  = test.copy()

    # Map: categorical columns → string
    dtype_map_mod = {col: 'str' for col in categorical_cols}
    # Assign dtypes in one go
    train_fixed = train_fixed.astype(dtype_map_mod)
    test_fixed  = test_fixed.astype(dtype_map_mod)

    #Gower function for cluster identification
    train = gower_distances.gower_closest_train(
    train=train_fixed,
    test=test_fixed,
    target_col=None,                   # or None if last column is the target
    cluster_size=CONFIG["cluster_size"]          # or omit → will be (sqrt(n))
    )

    # Map: categorical columns → string
    dtype_map_mod = {col: 'category' for col in categorical_cols}
    train = train.astype(dtype_map_mod)
    test  = test.astype(dtype_map_mod)

    # Encoding and scaling (if indicated at the beginning of the code) of variables
    X_train, X_test, y_train, y_test = encoder_scaler.freq_encode_and_scale(
    train_df=train,
    test_df=test,
    categorical_cols=categorical_cols,
    target_col="DV",
    do_scale=CONFIG["scalingY1N0"]          # set False if you don't want scaling
    )
    
    #Subset test and train to most important variables
    X_test = X_test.loc[:,final_predictors]
    X_train = X_train.loc[:,final_predictors]
    

    #MODELLING
    
    estimator, preds, metrics = models.fit_model(
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        model_name=CONFIG["model_name"],
        param_grid= CONFIG["param_grid"],
        search="grid",
        cv=CONFIG["cv_folds"],
        scoring=CONFIG["scoring"],
        return_metrics=True,
        y_test=y_test
    )

    prediction = preds

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, prediction)
    print(f'Mean Absolute Error on Test Set: {mae}')
    

    accuracy.at[i,'Sample Size']=len(train)
    accuracy.at[i,'Actual']=y_test
    accuracy.at[i,'MAE']= mae
    accuracy.at[i,'Predicted']= prediction
        
    print(i)


end = time.time()

print('It took ' + str(round((end-beginning)/60,1)) + ' minutes to train and test the model.')
naive_model_mae = abs(accuracy['Actual']-accuracy['Actual'].mean())
print('The mean absolute error was ' + str(round(accuracy['MAE'].mean(),1)) + ' points')
print('The standard deviation of the mean absolute error was ' + str(round(accuracy['MAE'].std(),1)) + ' points')
print('A naive model mean absolute error is ' + str(round(naive_model_mae.mean(),1)) + ' points')
print('A naive model standard deviation is ' + str(round(naive_model_mae.std(),1)) + ' points')

#Assess whether there is a statistical significant difference between the Naive and complex model
t_stat, p_val = stats.ttest_rel(accuracy['Actual']-accuracy['Actual'].mean(), accuracy['Actual']-accuracy['Predicted'])
print(f"t-statistic: {t_stat:.3f}")

print(f"two-tailed p-value: {p_val:.4f}")

