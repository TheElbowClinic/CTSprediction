
from tkinter.filedialog import askopenfilename
import pandas as pd
import time
import math
import random
import sklearn.tree
import sklearn.ensemble
import numpy as np

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

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data.isna(), cmap='Blues', cbar=True)
plt.show()

        #number of rows
data.shape[0]

data.dropna(subset=['GROC24Adjusted'],inplace=True) #Dropping rows that don't have GROC







# Missing datapoints
#Number of missing data, missing data plot, number of participant, and dataset dimensions (# rows and columns).

    #Missing data oversampled dataset
data.isna().sum().sum()

import matplotlib.pyplot as plt
import seaborn as sns

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











#Imputing missing data using the k-Nearest Neighbors approach
import numpy as np
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2, weights="uniform")
dd = imputer.fit_transform(data)

data_imputed = pd.DataFrame(dd, columns=data.columns)





# Missing datapoints
#Number of missing data, missing data plot, number of participant, and dataset dimensions (# rows and columns).

    #Missing data oversampled dataset
data_imputed.isna().sum().sum()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data_imputed.isna(), cmap='Blues', cbar=True)
plt.show()

        #number of rows
data_imputed.shape[0]

#data.dropna(inplace=True)

   










import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Correlation amongst variables
exclude_columns = ['sbj', '', '']

# Get columns to include in correlation
included_columns = [col for col in data.columns if col not in exclude_columns]

# Create a new DataFrame with only included columns
df_included = data[included_columns]

# Calculate the correlation matrix
correlation_matrix = df_included.corr()

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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









#dataFIN = data_imputed




#Gradient Boosting Machine (GBM) w/ Autotuning

from math import sqrt
import numpy as np
from scipy.special import logsumexp
import pandas as pd
import random
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
#from skopt import BayesSearchCV #For Bayesian selection of hyperparameters
from sklearn.ensemble import GradientBoostingRegressor #Gradient boosting machine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error  # Change the import
from sklearn.tree import DecisionTreeRegressor, plot_tree




data = dataFIN
data = data.rename(columns={'GROC24Adjusted': 'DV'})                   


scalingY1N0 = 0 # Determines whether data will be scaled (1) or not (0)

# Scaling of variables
if scalingY1N0 == 1:
    responder = data[data.columns[-1]]
    data_scaled = StandardScaler().fit_transform(data.drop(columns=[data.columns[-1]]))
    data = pd.DataFrame(data_scaled, columns=data.columns[:-1])
    data = data.reset_index(drop=True)
    responder = responder.reset_index(drop=True)
    data = pd.concat([data, responder], axis=1)
    

subset_pre = 10 # Number of predictors in the model created around a cluster


#accuracy = np.zeros((original.shape[0], 5 + (subset_pre*2) + 2))
accuracy = np.zeros((data.shape[0], 6))
accuracy = pd.DataFrame(accuracy)
accuracy.columns = ["Sample Size", "Actual", "MAE", "Predicted", "CorrectNoPain", "CorrectPain"]
accuracy = pd.concat([accuracy,pd.DataFrame(np.zeros((data.shape[0],80)))], axis = 1)




beginning = time.time()



for i in range(0,len(data)):
    print(i)

    test = data.iloc[i:i+1,:]
    train = data.drop(i, axis=0)
    
    # Calculate the Euclidean distance between the reference subject and all other subjects
    distances = cdist(train.iloc[:, :-1], test.iloc[:,:-1], metric='euclidean')
    distances = pd.DataFrame(distances)
    train_index = pd.DataFrame(train.index)
    distances = pd.concat([train_index,distances],axis=1)
    
    distances.columns = ["train_index","dist"]
    
    closest_subjects_indices = distances.sort_values("dist")
    closest_subjects_indices = closest_subjects_indices[0:20] #Size of cluster is sqrt of total sample size
    closest_subjects_indices = closest_subjects_indices['train_index']
    train = train.loc[closest_subjects_indices,:]
    
    
    
    
    
    
    # Calculate the correlation matrix
    correlation_matrix = train.corr()
    
    
    SigNoise = pd.DataFrame(0, index=[0], columns=train.columns[:-1])
    SigNoise.columns = train.columns[:-1]


    for z in range(SigNoise.shape[1]):
        x = abs(correlation_matrix.iloc[len(correlation_matrix)-1,z])
        SigNoise.iloc[:,z] = x
        del x
        
    
        
        
        
        
        
    
    # Ordering dataset columns based on SNR values
    SigNoise = SigNoise.sort_values(by=0, axis=1, ascending=False)
    
    #Substetting SigNoise to number of variables set above
    trial = SigNoise.iloc[:, 0:subset_pre]
    
    #Saving variables names
    PlotXnames_cluster = trial.columns
    PlotXnames_cluster = PlotXnames_cluster.insert(len(PlotXnames_cluster), 'DV')   
    
    test = test.loc[:,PlotXnames_cluster]
    train = train.loc[:,PlotXnames_cluster]   
    
    
    
    
    
    
    
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
        
    
    for pred in range(0,subset_pre): 
      accuracy.iloc[i,pred+6] = PlotXnames_cluster[pred]
      accuracy.iloc[i,pred+32] = trial.iloc[:,pred]
      
        
    print(i)



end = time.time()

print('It took ' + str(round((end-beginning)/60,1)) + ' minutes to train and test the model.')



naive_model_mae = abs(accuracy['Actual']-accuracy['Actual'].mean())

print('The mean absolute error was ' + str(round(accuracy['MAE'].mean(),1)) + ' points')
print('The standard deviation of the mean absolute error was ' + str(round(accuracy['MAE'].std(),1)) + ' points')
print('A naive model mean absolute error is ' + str(round(naive_model_mae.mean(),1)) + ' points')
print('A naive model standard deviation is ' + str(round(naive_model_mae.std(),1)) + ' points')













#Decision tree w/ Autotuning

from math import sqrt
import numpy as np
from scipy.special import logsumexp
import pandas as pd
import random
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
#from skopt import BayesSearchCV #For Bayesian selection of hyperparameters
from sklearn.ensemble import GradientBoostingRegressor #Gradient boosting machine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error  # Change the import
from sklearn.tree import DecisionTreeRegressor, plot_tree



data = dataFIN
data = data.rename(columns={'GROC24Adjusted': 'DV'})                   


scalingY1N0 = 0 # Determines whether data will be scaled (1) or not (0)

# Scaling of variables
if scalingY1N0 == 1:
    responder = data[data.columns[-1]]
    data_scaled = StandardScaler().fit_transform(data.drop(columns=[data.columns[-1]]))
    data = pd.DataFrame(data_scaled, columns=data.columns[:-1])
    data = data.reset_index(drop=True)
    responder = responder.reset_index(drop=True)
    data = pd.concat([data, responder], axis=1)
    

subset_pre = 10 # Number of predictors in the model created around a cluster


#accuracy = np.zeros((original.shape[0], 5 + (subset_pre*2) + 2))
accuracy = np.zeros((data.shape[0], 6))
accuracy = pd.DataFrame(accuracy)
accuracy.columns = ["Sample Size", "Actual", "MAE", "Predicted", "CorrectNoPain", "CorrectPain"]
accuracy = pd.concat([accuracy,pd.DataFrame(np.zeros((data.shape[0],80)))], axis = 1)




beginning = time.time()



for i in range(0,len(data)):
    print(i)

    test = data.iloc[i:i+1,:]
    train = data.drop(i, axis=0)
    
    # Calculate the Euclidean distance between the reference subject and all other subjects
    distances = cdist(train.iloc[:, :-1], test.iloc[:,:-1], metric='euclidean')
    distances = pd.DataFrame(distances)
    train_index = pd.DataFrame(train.index)
    distances = pd.concat([train_index,distances],axis=1)
    
    distances.columns = ["train_index","dist"]
    
    closest_subjects_indices = distances.sort_values("dist")
    closest_subjects_indices = closest_subjects_indices[0:20] #Size of cluster is sqrt of total sample size
    closest_subjects_indices = closest_subjects_indices['train_index']
    train = train.loc[closest_subjects_indices,:]
    
    
    
    
    
    
    # Calculate the correlation matrix
    correlation_matrix = train.corr()
    
    
    SigNoise = pd.DataFrame(0, index=[0], columns=train.columns[:-1])
    SigNoise.columns = train.columns[:-1]


    for z in range(SigNoise.shape[1]):
        x = abs(correlation_matrix.iloc[len(correlation_matrix)-1,z])
        SigNoise.iloc[:,z] = x
        del x
        
    
        
        
        
        
        
    
    # Ordering dataset columns based on SNR values
    SigNoise = SigNoise.sort_values(by=0, axis=1, ascending=False)
    
    #Substetting SigNoise to number of variables set above
    trial = SigNoise.iloc[:, 0:subset_pre]
    
    #Saving variables names
    PlotXnames_cluster = trial.columns
    PlotXnames_cluster = PlotXnames_cluster.insert(len(PlotXnames_cluster), 'DV')   
    
    test = test.loc[:,PlotXnames_cluster]
    train = train.loc[:,PlotXnames_cluster]   
    
    
    
    
    
    
    
    #MODELLING
    
    # Initialize the Decision Tree Regressor
    tree_regressor = DecisionTreeRegressor(random_state=42)

    # Define the parameter grid for the grid search
    param_grid = {
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'criterion': ['mae','mse']
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
        
    
    for pred in range(0,subset_pre): 
      accuracy.iloc[i,pred+6] = PlotXnames_cluster[pred]
      accuracy.iloc[i,pred+32] = trial.iloc[:,pred]
      
        
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

data = data.rename(columns={'CON1EXP2': 'Intervention'})
data = data.rename(columns={'SEXF1M0': 'Sex'})
data = data.rename(columns={'AGE': 'Age'})
data = data.rename(columns={'HAND_AFFECTEDR1L2B3': 'Affected side'})
data = data.rename(columns={'TIME_SINCE_SYMPTOM_ONSETyrs': 'Symptoms duration (yrs)'})
data = data.rename(columns={'HANDEDR1L0': 'Handedness'})
data = data.rename(columns={'unemp_ret0Office1Sales_service2Manual3': 'Employment status'})
data = data.rename(columns={'BCTQSSS0': 'BCTQ symptoms scale'})
data = data.rename(columns={'NCS_G': 'NCS'})
data = data.rename(columns={'PSURGY1N0': 'Surgical expectation'})
data = data.rename(columns={'spread0none_0forearm_1upper_2neck_3': 'Symptoms widespread'})
data = data.rename(columns={'GROC6Adjusted': 'GROC 6 weeks'})
data = data.rename(columns={'DV': 'GROC 24 weeks'})



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Correlation amongst variables
#exclude_columns = ['Sbj', 'class']
exclude_columns = []

# Get columns to include in correlation
included_columns = [col for col in data.columns if col not in exclude_columns]

# Create a new DataFrame with only included columns
df_included = data[included_columns]

# Calculate the correlation matrix
correlation_matrix = df_included.corr()

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


    #Creates matrix and then dataframe
SigNoise = np.zeros((1, df_included.shape[1] - 1))
SigNoise = pd.DataFrame(SigNoise)
SigNoise.columns = df_included.columns[0:12]

for i in range(SigNoise.shape[1]):
    x = abs(correlation_matrix.iloc[12,i])
    SigNoise.iloc[0, i] = x
del i


# Ordering data_imputed_over columns based on SNR values
SigNoise = SigNoise.sort_values(by=0, axis=1, ascending=False)

    #Substetting SigNoise to 50 variables
trial = SigNoise.iloc[:, :12]

    #Saving variables names
PlotXnames = trial.columns


y = trial.iloc[0, :].values #Absolute values

    # Plot CORRELATION SNR values.



# Line plot
plt.figure(dpi=600) # Set the resolution of the plot to 300 dpi
plt.plot(y,)
plt.xlabel('')
plt.ylabel('Correlation value')
plt.xticks(np.arange(12), PlotXnames, rotation=90, fontsize=6)
#plt.axvline(x=17, color='blue')
plt.show()



#Bar plot
plt.figure(dpi=600)
plt.bar(np.arange(12), height=y, align='center', alpha=0.9)
plt.xlabel('')
plt.ylabel('Correlation value')
plt.xticks(np.arange(12), PlotXnames, rotation=90, fontsize=6)
#plt.axvline(x=17, color='blue')
plt.show()


#Print the top 10 variables based on SNR
print(trial.columns[0:10])











