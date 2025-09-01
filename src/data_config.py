
# ────────────────────── CONFIG START ──────────────────────
columns_to_retain = ['CON1EXP2', 'SEXF1M0', 'AGE', 'HANDEDR1L0',
                     'HAND_AFFECTEDR1L2B3', 'TIME_SINCE_SYMPTOM_ONSETyrs',
                     'unemp_ret0Office1Sales_service2Manual3', 'BCTQSSS0', 'NCS_G',
                     'PSURGY1N0', 'spread0none_0forearm_1upper_2neck_3','GROC6Adjusted',
                     'GROC24Adjusted']

categorical_cols = ['CON1EXP2', 'SEXF1M0', 'HANDEDR1L0', 'HAND_AFFECTEDR1L2B3',
                    'unemp_ret0Office1Sales_service2Manual3', 'PSURGY1N0',
                    'spread0none_0forearm_1upper_2neck_3']

# List of columns to check frequency counts for - NCS_G is considered a continuous variable for modelling but used as categorical for frequency check.
columns_to_check = ['CON1EXP2', 'SEXF1M0', 'HANDEDR1L0', 'HAND_AFFECTEDR1L2B3',
                    'unemp_ret0Office1Sales_service2Manual3', 'PSURGY1N0',
                    'spread0none_0forearm_1upper_2neck_3', 'NCS_G']

#Continuous columns
continuous_cols = [col for col in columns_to_retain if col not in categorical_cols + ['GROC24Adjusted']]

#Final_predictors below are the variables for the modelling. These were selected after the signal to noise ratio analysis (correlation).
final_predictors = ['GROC6Adjusted','BCTQSSS0','PSURGY1N0','CON1EXP2',
                      'NCS_G','HAND_AFFECTEDR1L2B3','SEXF1M0',
                      'spread0none_0forearm_1upper_2neck_3','HANDEDR1L0',
                      'unemp_ret0Office1Sales_service2Manual3']



# Feature-selection thresholds
CONFIG = {
    "numb_var_assess": 12, #This indicates how many variables we want to have in SNR
    "cluster_size": 25, #This indicates the size of the cluster for the personalised modelling
    "scalingY1N0": True, # Determines whether data will be scaled (True) or not (False) prior to modelling
    "subset_pre": 10, # Number of predictors in the model created around a cluster
    "random_state": 42,
    "model_name": "dt",           # can be "random_forest", "ridge", …
    "param_grid": None,               # leave None for the default grid
    "cv_folds": 2,
    "scoring": "neg_mean_absolute_error",
}

# ────────────────────── CONFIG END ──────────────────────