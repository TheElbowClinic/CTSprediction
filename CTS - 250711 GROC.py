from tkinter.filedialog import askopenfilename
import pandas as pd
import time
import math
import random
import sklearn.tree
import sklearn.ensemble
import numpy as np

#For plotting and missing data
import matplotlib.pyplot as plt
import seaborn as sns

#To tabulate data
from tabulate import tabulate

#To impute data
import miceforest as mf #To impute categorical and continuous variables

#To oversample
from imblearn.over_sampling import SMOTENC

#For modelling
from math import sqrt
from scipy.special import logsumexp
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import tensorflow as tf
from scipy.spatial.distance import cdist #For Euclidean distances
from skopt import BayesSearchCV #For Bayesian selection of hyperparameters
import gower #For distances when variables are categorical and continuos
from pygam import LinearGAM
from sklearn.ensemble import GradientBoostingRegressor #Gradient boosting machine
from sklearn.metrics import mean_absolute_error  # Change the import
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


# Set the seed for reproducibility
np.random.seed(123)
sklearn.utils.check_random_state(123)

filename = askopenfilename()
FULLdataset = pd.read_excel(io=filename) #For the oversamples

dataset1 = FULLdataset
print(dataset1.columns)
print(dataset1.info())
print(dataset1.describe())

dataset2 = dataset1

# Variables to retain within dataset

col_names = dataset2.columns

print(col_names)

columns_to_retain = ['CON1EXP2', 'SEXF1M0', 'AGE', 'HANDEDR1L0',
                     'HAND_AFFECTEDR1L2B3', 'TIME_SINCE_SYMPTOM_ONSETyrs',
                     'unemp_ret0Office1Sales_service2Manual3', 'BCTQSSS0', 'NCS_G',
                     'PSURGY1N0', 'spread0none_0forearm_1upper_2neck_3','GROC6Adjusted',
                     'GROC24Adjusted']

dataset2 = dataset2[columns_to_retain]

print(dataset2.columns)
print(dataset2.info())
print(dataset2.describe())

dataset = dataset2

data = dataset


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



# Count total missing values
missing_count = data.isna().sum().sum()
print("Total missing datapoints:", missing_count)

# Get dataset dimensions: number of rows and columns
n_rows, n_cols = data.shape
print("Dataset dimensions: {} rows x {} columns".format(n_rows, n_cols))

# Compute total number of cells in the dataset
total_cells = n_rows * n_cols

# Calculate percentage of missing data
missing_percentage = (missing_count / total_cells) * 100
print("Percentage of missing data: {:.2f}%".format(missing_percentage))



#MAKE CATEGORICAL VARIABLE CATEGORICAL
# List of categorical columns to round
categorical_cols = [
    'CON1EXP2', 'SEXF1M0', 'HANDEDR1L0', 'HAND_AFFECTEDR1L2B3',
    'unemp_ret0Office1Sales_service2Manual3', 'PSURGY1N0',
    'spread0none_0forearm_1upper_2neck_3'
]



# Convert each column to categorical
data[categorical_cols] = data[categorical_cols].astype('category')





#Imputing missing data using the MICE for categorical and continuous variables.
#from sklearn.impute import KNNImputer


# Reset index
data = data.reset_index(drop=True)
# Create a MICE kernel
kernel = mf.ImputationKernel(
    data,
    #num_datasets=1,         # Number of multiple imputations
    random_state=42
)
# Run MICE and specify number of datasets
kernel.mice(
    iterations=5,
    num_datasets=1
)

# Run the MICE imputation process with # of iterations
kernel.mice(5)

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




#Descriptive statistics

describe_df = data.describe()

# Display as Markdown table (Jupyter or export)
print(tabulate(describe_df, headers='keys', tablefmt='github'))



#Frequency table

# List of columns you want to check frequency counts for
columns_to_check = [
    'CON1EXP2', 'SEXF1M0', 'HANDEDR1L0', 'HAND_AFFECTEDR1L2B3',
    'unemp_ret0Office1Sales_service2Manual3', 'PSURGY1N0',
    'spread0none_0forearm_1upper_2neck_3', 'NCS_G'
]

# Loop through each column and compute value counts
for col in columns_to_check:
    print(f"\n### Frequency Table for '{col}':")
    freq_table = data[col].value_counts().reset_index()
    freq_table.columns = ['Value', 'Frequency']
    print(tabulate(freq_table, headers='keys', tablefmt='github'))










# Correlation amongst variables
exclude_columns = ['sbj', '', '']

# Get columns to include in correlation
included_columns = [col for col in data.columns if col not in exclude_columns]

# Create a new DataFrame with only included columns
df_included = data[included_columns]

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








#ENCODE CATEGORICAL FOR SNR (Frequency encoding)
# Identify categorical and continuous variables
continuous_cols = [col for col in data_imputed.columns if col not in categorical_cols + ['GROC24Adjusted']]

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





# Correlation amongst variables
exclude_columns = ['sbj', '', '']

# Get columns to include in correlation
included_columns = [col for col in data.columns if col not in exclude_columns]

# Create a new DataFrame with only included columns
df_included = data_imputed_encoded[included_columns]

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
numb_var_assess = 12 #This indicates how many variables we want to have in SNR

    #Creates matrix and then dataframe
SigNoise = np.zeros((1, df_included.shape[1]-1))
SigNoise = pd.DataFrame(SigNoise)
SigNoise.columns = df_included.columns[0:numb_var_assess]

for i in range(SigNoise.shape[1]):
    x = abs(correlation_matrix.iloc[len(correlation_matrix)-1,i])
    SigNoise.iloc[0, i] = x
del i


# Ordering data_imputed_over columns based on SNR values
SigNoise = SigNoise.sort_values(by=0, axis=1, ascending=False)

    #Substetting SigNoise to number of variables
trial = SigNoise.iloc[:, :numb_var_assess]

    #Saving variables names
PlotXnames = trial.columns


y = trial.iloc[0, :].values #Absolute values

    # Plot CORRELATION SNR values.


# Line plot
plt.figure(dpi=600) # Set the resolution of the plot to 300 dpi
plt.plot(y,)
plt.xlabel('')
plt.ylabel('Correlation value')
plt.xticks(np.arange(len(PlotXnames)), PlotXnames, rotation=45, fontsize=6)
#plt.axvline(x=17, color='blue')
plt.show()



#Bar plot
plt.figure(dpi=600)
plt.bar(np.arange(len(PlotXnames)), height=y, align='center', alpha=0.9)
plt.xlabel('')
plt.ylabel('Correlation value')
plt.xticks(np.arange(len(PlotXnames)), PlotXnames, rotation=45, fontsize=6)
#plt.axvline(x=17, color='blue')
plt.show()


#Print the top 10 variables based on SNR
print(trial.columns[0:len(PlotXnames)])

#Create new dataset which retains only most important predictors and dependent variable
nomi = PlotXnames.insert(len(PlotXnames), 'GROC24Adjusted') 
dataFIN = data_imputed[nomi]









#Linear regression

data = dataFIN
data = data.rename(columns={'GROC24Adjusted': 'DV'})                   
scalingY1N0 = 1 # Determines whether data will be scaled (1) or not (0)
subset_pre = 10 # Number of predictors in the model created around a cluster

accuracy = np.zeros((data.shape[0], 6))
accuracy = pd.DataFrame(accuracy)
accuracy.columns = ["Sample Size", "Actual", "MAE", "Predicted", "CorrectNoPain", "CorrectPain"]
accuracy = pd.concat([accuracy,pd.DataFrame(np.zeros((data.shape[0],80)))], axis = 1)

beginning = time.time()

for i in range(0,len(data)):
    print(i)

    test = data.iloc[i:i+1,:]
    train = data.drop(i, axis=0)
    
    # Convert all columns to standard NumPy types for the gower distances to be computed
    train_fixed = train.astype({col: 'float' for col in train.columns})
    test_fixed = test.astype({col: 'float' for col in test.columns})
    
    # Compute Gower distances between training and test data (excluding target columns)
    distances = gower.gower_matrix(train_fixed.iloc[:, :-1], test_fixed.iloc[:, :-1])
    #distances = cdist(train.iloc[:, :-1], test.iloc[:,:-1], metric='euclidean') #This only works when there are only continuous variables
    distances = pd.DataFrame(distances)
    train_index = pd.DataFrame(train.index)
    distances = pd.concat([train_index,distances],axis=1)
    
    distances.columns = ["train_index","dist"]
    
    closest_subjects_indices = distances.sort_values("dist")
    closest_subjects_indices = closest_subjects_indices[0:25] #Size of cluster is sqrt of total sample size
    closest_subjects_indices = closest_subjects_indices['train_index']
    train = train.loc[closest_subjects_indices,:]
    

# Scaling of variables
    if scalingY1N0 == 1:
        # Identify categorical and continuous variables
        continuous_cols = [col for col in train.columns if col not in categorical_cols + ['DV']]
        
        # Store the responder column
        responder = train['DV']
        responder_test = test['DV']
        
        # Store frequency mappings per column
        freq_maps = {}

        # Frequency encode categorical variables in training data
        train_encoded = train.copy()
        for col in categorical_cols:
            # Calculate frequency of each category
            freq = train[col].astype(str).value_counts(normalize=True)
            #print(freq)
            # Store it for test mapping
            freq_maps[col] = freq 
            # Replace categories with their frequencies
            train_encoded[col] = train[col].astype(str).map(freq_maps[col]).astype(float)
        
        # Apply the same encoding to test data
        test_encoded = test.copy()
        for col in categorical_cols:
            test_encoded[col] = test[col].astype(str).map(freq_maps[col]).fillna(0).astype(float)
        
        # Scale all features (now including encoded categorical variables)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_encoded.drop(columns=['DV']))
        test_scaled = scaler.transform(test_encoded.drop(columns=['DV']))
        
        # Convert back to DataFrame
        train = pd.DataFrame(train_scaled, columns=train_encoded.columns[:-1])
        test = pd.DataFrame(test_scaled, columns=test_encoded.columns[:-1])
        
        # Reset indices
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        responder = responder.reset_index(drop=True)
        responder_test = responder_test.reset_index(drop=True)
        
        # Add back responder columns
        train = pd.concat([train, responder], axis=1)
        test = pd.concat([test, responder_test], axis=1)    
    
    
    
    
    #Porcello_porco below are the variables for the modelling
    porcello_porco = ['GROC6Adjusted','BCTQSSS0','PSURGY1N0','CON1EXP2',
                      'NCS_G','HAND_AFFECTEDR1L2B3','SEXF1M0',
                      'spread0none_0forearm_1upper_2neck_3','HANDEDR1L0',
                      'unemp_ret0Office1Sales_service2Manual3','DV']
    test = test.loc[:,porcello_porco]
    train = train.loc[:,porcello_porco]
    
    
    #MODELLING
    
    # Initialize the Gradient Boosting Regressor
    linear_regressor = LinearRegression()

    # Fit the model to the training data using grid search
    linear_regressor.fit(train.iloc[:,:-1], train["DV"])


    # Make predictions on the test set using the best model
    prediction = linear_regressor.predict(test.iloc[:,:-1])

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(test['DV'], prediction)
    print(f'Mean Absolute Error on Test Set: {mae}')
    

    accuracy.at[i,'Sample Size']=len(train)
    accuracy.at[i,'Actual']=test['DV']
    accuracy.at[i,'MAE']= mae
    accuracy.at[i,'Predicted']= prediction #2 if model.best_estimator_.predict_proba(test.iloc[:,:-1])[0][0] < 0.6 else 1
        
    print(i)



end = time.time()

print('It took ' + str(round((end-beginning)/60,1)) + ' minutes to train and test the model.')



naive_model_mae = abs(accuracy['Actual']-accuracy['Actual'].mean())

print('The mean absolute error was ' + str(round(accuracy['MAE'].mean(),1)) + ' points')
print('The standard deviation of the mean absolute error was ' + str(round(accuracy['MAE'].std(),1)) + ' points')
print('A naive model mean absolute error is ' + str(round(naive_model_mae.mean(),1)) + ' points')
print('A naive model standard deviation is ' + str(round(naive_model_mae.std(),1)) + ' points')














#Ridge regression w/ Autotuning

data = dataFIN
data = data.rename(columns={'GROC24Adjusted': 'DV'})                   
scalingY1N0 = 1 # Determines whether data will be scaled (1) or not (0)
subset_pre = 10 # Number of predictors in the model created around a cluster

accuracy = np.zeros((data.shape[0], 6))
accuracy = pd.DataFrame(accuracy)
accuracy.columns = ["Sample Size", "Actual", "MAE", "Predicted", "CorrectNoPain", "CorrectPain"]
accuracy = pd.concat([accuracy,pd.DataFrame(np.zeros((data.shape[0],80)))], axis = 1)

beginning = time.time()


for i in range(0,len(data)):
    print(i)

    test = data.iloc[i:i+1,:]
    train = data.drop(i, axis=0)
    
    # Convert all columns to standard NumPy types for the gower distances to be computed
    train_fixed = train.astype({col: 'float' for col in train.columns})
    test_fixed = test.astype({col: 'float' for col in test.columns})
    
    # Compute Gower distances between training and test data (excluding target columns)
    distances = gower.gower_matrix(train_fixed.iloc[:, :-1], test_fixed.iloc[:, :-1])
    #distances = cdist(train.iloc[:, :-1], test.iloc[:,:-1], metric='euclidean') #This only works when there are only continuous variables
    distances = pd.DataFrame(distances)
    train_index = pd.DataFrame(train.index)
    distances = pd.concat([train_index,distances],axis=1)
    
    distances.columns = ["train_index","dist"]
    
    closest_subjects_indices = distances.sort_values("dist")
    closest_subjects_indices = closest_subjects_indices[0:25] #Size of cluster is sqrt of total sample size
    closest_subjects_indices = closest_subjects_indices['train_index']
    train = train.loc[closest_subjects_indices,:]
    

# Scaling of variables
    if scalingY1N0 == 1:
        # Identify categorical and continuous variables
        continuous_cols = [col for col in train.columns if col not in categorical_cols + ['DV']]
        
        # Store the responder column
        responder = train['DV']
        responder_test = test['DV']
        
        # Store frequency mappings per column
        freq_maps = {}

        # Frequency encode categorical variables in training data
        train_encoded = train.copy()
        for col in categorical_cols:
            # Calculate frequency of each category
            freq = train[col].astype(str).value_counts(normalize=True)
            #print(freq)
            # Store it for test mapping
            freq_maps[col] = freq 
            # Replace categories with their frequencies
            train_encoded[col] = train[col].astype(str).map(freq_maps[col]).astype(float)
        
        # Apply the same encoding to test data
        test_encoded = test.copy()
        for col in categorical_cols:
            test_encoded[col] = test[col].astype(str).map(freq_maps[col]).fillna(0).astype(float)
        
        # Scale all features (now including encoded categorical variables)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_encoded.drop(columns=['DV']))
        test_scaled = scaler.transform(test_encoded.drop(columns=['DV']))
        
        # Convert back to DataFrame
        train = pd.DataFrame(train_scaled, columns=train_encoded.columns[:-1])
        test = pd.DataFrame(test_scaled, columns=test_encoded.columns[:-1])
        
        # Reset indices
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        responder = responder.reset_index(drop=True)
        responder_test = responder_test.reset_index(drop=True)
        
        # Add back responder columns
        train = pd.concat([train, responder], axis=1)
        test = pd.concat([test, responder_test], axis=1)    
    
    
    
    
    #Porcello_porco below are the variables for the modelling
    porcello_porco = ['GROC6Adjusted','BCTQSSS0','PSURGY1N0','CON1EXP2',
                      'NCS_G','HAND_AFFECTEDR1L2B3','SEXF1M0',
                      'spread0none_0forearm_1upper_2neck_3','HANDEDR1L0',
                      'unemp_ret0Office1Sales_service2Manual3','DV']
    test = test.loc[:,porcello_porco]
    train = train.loc[:,porcello_porco]
    
    
    #MODELLING
    
    # Initialize the Gradient Boosting Regressor
    ridge_regressor = Ridge()

    # Define the parameter grid for the grid search
    param_grid = {
        'alpha': np.logspace(-2, 2, 10)  # 0.01 to 100
    }

    # Initialize GridSearchCV
    model = GridSearchCV(ridge_regressor, param_grid, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit the model to the training data using grid search
    model.fit(train.iloc[:,:-1], train["DV"])

    # Get the best parameters from the grid search
    best_params = model.best_params_
    print(f'Best Parameters: {best_params}')

    # Make predictions on the test set using the best model
    best_estimator = model.best_estimator_
    prediction = best_estimator.predict(test.iloc[:,:-1])

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(test['DV'], prediction)
    print(f'Mean Absolute Error on Test Set: {mae}')

    accuracy.at[i,'Sample Size']=len(train)
    accuracy.at[i,'Actual']=test['DV']
    accuracy.at[i,'MAE']= mae
    accuracy.at[i,'Predicted']= prediction #2 if model.best_estimator_.predict_proba(test.iloc[:,:-1])[0][0] < 0.6 else 1
        
    print(i)



end = time.time()

print('It took ' + str(round((end-beginning)/60,1)) + ' minutes to train and test the model.')



naive_model_mae = abs(accuracy['Actual']-accuracy['Actual'].mean())

print('The mean absolute error was ' + str(round(accuracy['MAE'].mean(),1)) + ' points')
print('The standard deviation of the mean absolute error was ' + str(round(accuracy['MAE'].std(),1)) + ' points')
print('A naive model mean absolute error is ' + str(round(naive_model_mae.mean(),1)) + ' points')
print('A naive model standard deviation is ' + str(round(naive_model_mae.std(),1)) + ' points')










#Support vector regression w/ Autotuning

data = dataFIN
data = data.rename(columns={'GROC24Adjusted': 'DV'})                   
scalingY1N0 = 1 # Determines whether data will be scaled (1) or not (0)
subset_pre = 10 # Number of predictors in the model created around a cluster
accuracy = np.zeros((data.shape[0], 6))
accuracy = pd.DataFrame(accuracy)
accuracy.columns = ["Sample Size", "Actual", "MAE", "Predicted", "CorrectNoPain", "CorrectPain"]
accuracy = pd.concat([accuracy,pd.DataFrame(np.zeros((data.shape[0],80)))], axis = 1)

beginning = time.time()


for i in range(0,len(data)):
    print(i)

    test = data.iloc[i:i+1,:]
    train = data.drop(i, axis=0)
    
    # Convert all columns to standard NumPy types for the gower distances to be computed
    train_fixed = train.astype({col: 'float' for col in train.columns})
    test_fixed = test.astype({col: 'float' for col in test.columns})
    
    # Compute Gower distances between training and test data (excluding target columns)
    distances = gower.gower_matrix(train_fixed.iloc[:, :-1], test_fixed.iloc[:, :-1])
    #distances = cdist(train.iloc[:, :-1], test.iloc[:,:-1], metric='euclidean') #This only works when there are only continuous variables
    distances = pd.DataFrame(distances)
    train_index = pd.DataFrame(train.index)
    distances = pd.concat([train_index,distances],axis=1)
    
    distances.columns = ["train_index","dist"]
    
    closest_subjects_indices = distances.sort_values("dist")
    closest_subjects_indices = closest_subjects_indices[0:25] #Size of cluster is sqrt of total sample size
    closest_subjects_indices = closest_subjects_indices['train_index']
    train = train.loc[closest_subjects_indices,:]
    

# Scaling of variables
    if scalingY1N0 == 1:
        # Identify categorical and continuous variables
        continuous_cols = [col for col in train.columns if col not in categorical_cols + ['DV']]
        
        # Store the responder column
        responder = train['DV']
        responder_test = test['DV']
        
        # Store frequency mappings per column
        freq_maps = {}

        # Frequency encode categorical variables in training data
        train_encoded = train.copy()
        for col in categorical_cols:
            # Calculate frequency of each category
            freq = train[col].astype(str).value_counts(normalize=True)
            #print(freq)
            # Store it for test mapping
            freq_maps[col] = freq 
            # Replace categories with their frequencies
            train_encoded[col] = train[col].astype(str).map(freq_maps[col]).astype(float)
        
        # Apply the same encoding to test data
        test_encoded = test.copy()
        for col in categorical_cols:
            test_encoded[col] = test[col].astype(str).map(freq_maps[col]).fillna(0).astype(float)
        
        # Scale all features (now including encoded categorical variables)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_encoded.drop(columns=['DV']))
        test_scaled = scaler.transform(test_encoded.drop(columns=['DV']))
        
        # Convert back to DataFrame
        train = pd.DataFrame(train_scaled, columns=train_encoded.columns[:-1])
        test = pd.DataFrame(test_scaled, columns=test_encoded.columns[:-1])
        
        # Reset indices
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        responder = responder.reset_index(drop=True)
        responder_test = responder_test.reset_index(drop=True)
        
        # Add back responder columns
        train = pd.concat([train, responder], axis=1)
        test = pd.concat([test, responder_test], axis=1)    
    
    
    
    
    #Porcello_porco below are the variables for the modelling
    porcello_porco = ['GROC6Adjusted','BCTQSSS0','PSURGY1N0','CON1EXP2',
                      'NCS_G','HAND_AFFECTEDR1L2B3','SEXF1M0',
                      'spread0none_0forearm_1upper_2neck_3','HANDEDR1L0',
                      'unemp_ret0Office1Sales_service2Manual3','DV']
    test = test.loc[:,porcello_porco]
    train = train.loc[:,porcello_porco]
    
    
    #MODELLING
    
    # Initialize the Gradient Boosting Regressor
    SVR_regressor = SVR()

    # Define the parameter grid for the grid search
    param_grid = {
    'C': np.logspace(-2, 2, 5),  # 5 values
    'epsilon': [0.1, 0.2],       # Epsilon parameter for the margin of tolerance
    'kernel': ['linear', 'rbf'], # Kernel types to try: linear and RBF
    }

    # Initialize GridSearchCV
    model = GridSearchCV(SVR_regressor, param_grid, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit the model to the training data using grid search
    model.fit(train.iloc[:,:-1], train["DV"])

    # Get the best parameters from the grid search
    best_params = model.best_params_
    print(f'Best Parameters: {best_params}')

    # Make predictions on the test set using the best model
    best_estimator = model.best_estimator_
    prediction = best_estimator.predict(test.iloc[:,:-1])

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(test['DV'], prediction)
    print(f'Mean Absolute Error on Test Set: {mae}')

    accuracy.at[i,'Sample Size']=len(train)
    accuracy.at[i,'Actual']=test['DV']
    accuracy.at[i,'MAE']= mae
    accuracy.at[i,'Predicted']= prediction #2 if model.best_estimator_.predict_proba(test.iloc[:,:-1])[0][0] < 0.6 else 1

    print(i)



end = time.time()

print('It took ' + str(round((end-beginning)/60,1)) + ' minutes to train and test the model.')



naive_model_mae = abs(accuracy['Actual']-accuracy['Actual'].mean())

print('The mean absolute error was ' + str(round(accuracy['MAE'].mean(),1)) + ' points')
print('The standard deviation of the mean absolute error was ' + str(round(accuracy['MAE'].std(),1)) + ' points')
print('A naive model mean absolute error is ' + str(round(naive_model_mae.mean(),1)) + ' points')
print('A naive model standard deviation is ' + str(round(naive_model_mae.std(),1)) + ' points')










#Gradient Boosting Machine (GBM) w/ Autotuning

data = dataFIN
data = data.rename(columns={'GROC24Adjusted': 'DV'})                   
scalingY1N0 = 1 # Determines whether data will be scaled (1) or not (0)
subset_pre = 10 # Number of predictors in the model created around a cluster
accuracy = np.zeros((data.shape[0], 6))
accuracy = pd.DataFrame(accuracy)
accuracy.columns = ["Sample Size", "Actual", "MAE", "Predicted", "CorrectNoPain", "CorrectPain"]
accuracy = pd.concat([accuracy,pd.DataFrame(np.zeros((data.shape[0],80)))], axis = 1)

beginning = time.time()


for i in range(0,len(data)):
    print(i)

    test = data.iloc[i:i+1,:]
    train = data.drop(i, axis=0)
    
    # Convert all columns to standard NumPy types for the gower distances to be computed
    train_fixed = train.astype({col: 'float' for col in train.columns})
    test_fixed = test.astype({col: 'float' for col in test.columns})
    
    # Compute Gower distances between training and test data (excluding target columns)
    distances = gower.gower_matrix(train_fixed.iloc[:, :-1], test_fixed.iloc[:, :-1])
    #distances = cdist(train.iloc[:, :-1], test.iloc[:,:-1], metric='euclidean') #This only works when there are only continuous variables
    distances = pd.DataFrame(distances)
    train_index = pd.DataFrame(train.index)
    distances = pd.concat([train_index,distances],axis=1)
    
    distances.columns = ["train_index","dist"]
    
    closest_subjects_indices = distances.sort_values("dist")
    closest_subjects_indices = closest_subjects_indices[0:25] #Size of cluster is sqrt of total sample size
    closest_subjects_indices = closest_subjects_indices['train_index']
    train = train.loc[closest_subjects_indices,:]
    

# Scaling of variables
    if scalingY1N0 == 1:
        # Identify categorical and continuous variables
        continuous_cols = [col for col in train.columns if col not in categorical_cols + ['DV']]
        
        # Store the responder column
        responder = train['DV']
        responder_test = test['DV']
        
        # Store frequency mappings per column
        freq_maps = {}

        # Frequency encode categorical variables in training data
        train_encoded = train.copy()
        for col in categorical_cols:
            # Calculate frequency of each category
            freq = train[col].astype(str).value_counts(normalize=True)
            #print(freq)
            # Store it for test mapping
            freq_maps[col] = freq 
            # Replace categories with their frequencies
            train_encoded[col] = train[col].astype(str).map(freq_maps[col]).astype(float)
        
        # Apply the same encoding to test data
        test_encoded = test.copy()
        for col in categorical_cols:
            test_encoded[col] = test[col].astype(str).map(freq_maps[col]).fillna(0).astype(float)
        
        # Scale all features (now including encoded categorical variables)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_encoded.drop(columns=['DV']))
        test_scaled = scaler.transform(test_encoded.drop(columns=['DV']))
        
        # Convert back to DataFrame
        train = pd.DataFrame(train_scaled, columns=train_encoded.columns[:-1])
        test = pd.DataFrame(test_scaled, columns=test_encoded.columns[:-1])
        
        # Reset indices
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        responder = responder.reset_index(drop=True)
        responder_test = responder_test.reset_index(drop=True)
        
        # Add back responder columns
        train = pd.concat([train, responder], axis=1)
        test = pd.concat([test, responder_test], axis=1)    
    
    
    
    
    #Porcello_porco below are the variables for the modelling
    porcello_porco = ['GROC6Adjusted','BCTQSSS0','PSURGY1N0','CON1EXP2',
                      'NCS_G','HAND_AFFECTEDR1L2B3','SEXF1M0',
                      'spread0none_0forearm_1upper_2neck_3','HANDEDR1L0',
                      'unemp_ret0Office1Sales_service2Manual3','DV']
    test = test.loc[:,porcello_porco]
    train = train.loc[:,porcello_porco]
    
    
    #MODELLING
    
    # Initialize the Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor(random_state=42)

    # Define the parameter grid for the grid search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
    }

    # Initialize GridSearchCV
    model = GridSearchCV(gb_regressor, param_grid, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit the model to the training data using grid search
    model.fit(train.iloc[:,:-1], train["DV"])

    # Get the best parameters from the grid search
    best_params = model.best_params_
    print(f'Best Parameters: {best_params}')

    # Make predictions on the test set using the best model
    best_estimator = model.best_estimator_
    prediction = best_estimator.predict(test.iloc[:,:-1])

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(test['DV'], prediction)
    print(f'Mean Absolute Error on Test Set: {mae}')

    accuracy.at[i,'Sample Size']=len(train)
    accuracy.at[i,'Actual']=test['DV']
    accuracy.at[i,'MAE']= mae
    accuracy.at[i,'Predicted']= prediction #2 if model.best_estimator_.predict_proba(test.iloc[:,:-1])[0][0] < 0.6 else 1

    print(i)



end = time.time()

print('It took ' + str(round((end-beginning)/60,1)) + ' minutes to train and test the model.')



naive_model_mae = abs(accuracy['Actual']-accuracy['Actual'].mean())

print('The mean absolute error was ' + str(round(accuracy['MAE'].mean(),1)) + ' points')
print('The standard deviation of the mean absolute error was ' + str(round(accuracy['MAE'].std(),1)) + ' points')
print('A naive model mean absolute error is ' + str(round(naive_model_mae.mean(),1)) + ' points')
print('A naive model standard deviation is ' + str(round(naive_model_mae.std(),1)) + ' points')













#Decision tree w/ Autotuning

data = dataFIN
data = data.rename(columns={'GROC24Adjusted': 'DV'})                   
scalingY1N0 = 1 # Determines whether data will be scaled (1) or not (0)
subset_pre = 10 # Number of predictors in the model created around a cluster
accuracy = np.zeros((data.shape[0], 6))
accuracy = pd.DataFrame(accuracy)
accuracy.columns = ["Sample Size", "Actual", "MAE", "Predicted", "CorrectNoPain", "CorrectPain"]
accuracy = pd.concat([accuracy,pd.DataFrame(np.zeros((data.shape[0],80)))], axis = 1)

beginning = time.time()


for i in range(0,len(data)):
    print(i)

    test = data.iloc[i:i+1,:]
    train = data.drop(i, axis=0)
    
    # Convert all columns to standard NumPy types for the gower distances to be computed
    train_fixed = train.astype({col: 'float' for col in train.columns})
    test_fixed = test.astype({col: 'float' for col in test.columns})
    
    # Compute Gower distances between training and test data (excluding target columns)
    distances = gower.gower_matrix(train_fixed.iloc[:, :-1], test_fixed.iloc[:, :-1])
    #distances = cdist(train.iloc[:, :-1], test.iloc[:,:-1], metric='euclidean') #This only works when there are only continuous variables
    distances = pd.DataFrame(distances)
    train_index = pd.DataFrame(train.index)
    distances = pd.concat([train_index,distances],axis=1)
    
    distances.columns = ["train_index","dist"]
    
    closest_subjects_indices = distances.sort_values("dist")
    closest_subjects_indices = closest_subjects_indices[0:25] #Size of cluster is sqrt of total sample size
    closest_subjects_indices = closest_subjects_indices['train_index']
    train = train.loc[closest_subjects_indices,:]
    

# Scaling of variables
    if scalingY1N0 == 1:
        # Identify categorical and continuous variables
        continuous_cols = [col for col in train.columns if col not in categorical_cols + ['DV']]
        
        # Store the responder column
        responder = train['DV']
        responder_test = test['DV']
        
        # Store frequency mappings per column
        freq_maps = {}

        # Frequency encode categorical variables in training data
        train_encoded = train.copy()
        for col in categorical_cols:
            # Calculate frequency of each category
            freq = train[col].astype(str).value_counts(normalize=True)
            #print(freq)
            # Store it for test mapping
            freq_maps[col] = freq 
            # Replace categories with their frequencies
            train_encoded[col] = train[col].astype(str).map(freq_maps[col]).astype(float)
        
        # Apply the same encoding to test data
        test_encoded = test.copy()
        for col in categorical_cols:
            test_encoded[col] = test[col].astype(str).map(freq_maps[col]).fillna(0).astype(float)
        
        # Scale all features (now including encoded categorical variables)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_encoded.drop(columns=['DV']))
        test_scaled = scaler.transform(test_encoded.drop(columns=['DV']))
        
        # Convert back to DataFrame
        train = pd.DataFrame(train_scaled, columns=train_encoded.columns[:-1])
        test = pd.DataFrame(test_scaled, columns=test_encoded.columns[:-1])
        
        # Reset indices
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        responder = responder.reset_index(drop=True)
        responder_test = responder_test.reset_index(drop=True)
        
        # Add back responder columns
        train = pd.concat([train, responder], axis=1)
        test = pd.concat([test, responder_test], axis=1)    
    
    
    
    
    #Porcello_porco below are the variables for the modelling
    porcello_porco = ['GROC6Adjusted','BCTQSSS0','PSURGY1N0','CON1EXP2',
                      'NCS_G','HAND_AFFECTEDR1L2B3','SEXF1M0',
                      'spread0none_0forearm_1upper_2neck_3','HANDEDR1L0',
                      'unemp_ret0Office1Sales_service2Manual3','DV']
    test = test.loc[:,porcello_porco]
    train = train.loc[:,porcello_porco]
    
    
    
    #MODELLING
    
    # Initialize the Decision Tree Regressor
    tree_regressor = DecisionTreeRegressor(random_state=42)

    # Define the parameter grid for the grid search
    param_grid = {
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'criterion': ['absolute_error', 'squared_error']
    }
    
    # Initialize GridSearchCV
    model = GridSearchCV(tree_regressor, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit the model to the training data using grid search
    model.fit(train.iloc[:,:-1], train["DV"])

    # Get the best parameters from the grid search
    best_params = model.best_params_
    print(f'Best Parameters: {best_params}')

    # Make predictions on the test set using the best model
    best_estimator = model.best_estimator_
    prediction = best_estimator.predict(test.iloc[:,:-1])

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(test['DV'], prediction)
    print(f'Mean Absolute Error on Test Set: {mae}')

    accuracy.at[i,'Sample Size']=len(train)
    accuracy.at[i,'Actual']=test['DV']
    accuracy.at[i,'MAE']= mae
    accuracy.at[i,'Predicted']= prediction #2 if model.best_estimator_.predict_proba(test.iloc[:,:-1])[0][0] < 0.6 else 1
 
    print(i)



end = time.time()

print('It took ' + str(round((end-beginning)/60,1)) + ' minutes to train and test the model.')



naive_model_mae = abs(accuracy['Actual']-accuracy['Actual'].mean())

print('The mean mean absolute error was ' + str(round(accuracy['MAE'].mean(),1)) + ' degrees')
print('The standard deviation of the mean absolute error was ' + str(round(accuracy['MAE'].std(),1)) + ' degrees')
print('A naive model mean absolute erro is ' + str(round(naive_model_mae.mean(),1)) + ' degrees')
print('A naive mode standard deviation is ' + str(round(naive_model_mae.std(),1)) + ' degrees')
























# Code to print the pictures with legible names



#Rename dataset variables

data_imputed = data_imputed.rename(columns={'CON1EXP2': 'Intervention'})
data_imputed = data_imputed.rename(columns={'SEXF1M0': 'Sex'})
data_imputed = data_imputed.rename(columns={'AGE': 'Age'})
data_imputed = data_imputed.rename(columns={'HAND_AFFECTEDR1L2B3': 'Affected side'})
data_imputed = data_imputed.rename(columns={'TIME_SINCE_SYMPTOM_ONSETyrs': 'Symptoms duration (yrs)'})
data_imputed = data_imputed.rename(columns={'HANDEDR1L0': 'Handedness'})
data_imputed = data_imputed.rename(columns={'unemp_ret0Office1Sales_service2Manual3': 'Employment status'})
data_imputed = data_imputed.rename(columns={'BCTQSSS0': 'BCTQ symptoms scale'})
data_imputed = data_imputed.rename(columns={'NCS_G': 'NCS'})
data_imputed = data_imputed.rename(columns={'PSURGY1N0': 'Surgical expectation'})
data_imputed = data_imputed.rename(columns={'spread0none_0forearm_1upper_2neck_3': 'Symptoms widespread'})
data_imputed = data_imputed.rename(columns={'GROC6Adjusted': 'GROC 6 weeks'})
data_imputed = data_imputed.rename(columns={'DV': 'GROC 24 weeks'})





# List of categorical columns to round
categorical_cols = [
    'Intervention', 'Sex', 'Handedness', 'Affected side',
    'Employment status', 'Surgical expectation',
    'Symptoms widespread'
]






#ENCODE CATEGORICAL FOR SNR (Frequency encoding)
# Identify categorical and continuous variables
continuous_cols = [col for col in data_imputed.columns if col not in categorical_cols + ['GROC24Adjusted']]

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








# Correlation amongst variables
exclude_columns = ['sbj', '', '']

# Get columns to include in correlation
included_columns = [col for col in data_imputed.columns if col not in exclude_columns]

# Create a new DataFrame with only included columns
df_included = data_imputed_encoded[included_columns]

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


numb_var_assess = 12 #This indicates how many variables we want to have in SNR

    #Creates matrix and then dataframe
SigNoise = np.zeros((1, df_included.shape[1]-1))
SigNoise = pd.DataFrame(SigNoise)
SigNoise.columns = df_included.columns[0:numb_var_assess]

for i in range(SigNoise.shape[1]):
    x = abs(correlation_matrix.iloc[len(correlation_matrix)-1,i])
    SigNoise.iloc[0, i] = x
del i


# Ordering data_imputed_over columns based on SNR values
SigNoise = SigNoise.sort_values(by=0, axis=1, ascending=False)

    #Substetting SigNoise to number of variables
trial = SigNoise.iloc[:, :numb_var_assess]

    #Saving variables names
PlotXnames = trial.columns


y = trial.iloc[0, :].values #Absolute values

    # Plot CORRELATION SNR values.


# Line plot
plt.figure(dpi=600) # Set the resolution of the plot to 300 dpi
plt.plot(y,)
plt.xlabel('')
plt.ylabel('Correlation value')
plt.xticks(np.arange(len(PlotXnames)), PlotXnames, rotation=90, fontsize=6)
#plt.axvline(x=17, color='blue')
plt.show()



#Bar plot
plt.figure(dpi=600)
plt.bar(np.arange(len(PlotXnames)), height=y, align='center', alpha=0.9)
plt.xlabel('')
plt.ylabel('Correlation value')
plt.xticks(np.arange(len(PlotXnames)), PlotXnames, rotation=90, fontsize=6)
#plt.axvline(x=17, color='blue')
plt.show()


#Print the top 10 variables based on SNR
print(trial.columns[0:len(PlotXnames)])